#!/bin/bash
# ============================================================================
# upload_to_hdfs.sh
# Scalable Cross-City AQI Forecasting — HDFS Data Upload Script
# ============================================================================
#
# PREREQUISITES — Hadoop Installation on Windows
# ================================================
#
# 1. Download Hadoop 3.3.x binary from:
#       https://hadoop.apache.org/releases.html
#    Extract to e.g. C:\hadoop
#
# 2. Download winutils.exe for your Hadoop version from:
#       https://github.com/cdarlint/winutils
#    Place winutils.exe in C:\hadoop\bin\
#
# 3. Set environment variables (System Properties → Environment Variables):
#       HADOOP_HOME  = C:\hadoop
#       JAVA_HOME    = C:\Program Files\Java\jdk-11  (or your JDK path)
#       Add to PATH:  %HADOOP_HOME%\bin ; %HADOOP_HOME%\sbin
#
# 4. Configure core-site.xml (C:\hadoop\etc\hadoop\core-site.xml):
#       <configuration>
#         <property>
#           <name>fs.defaultFS</name>
#           <value>hdfs://localhost:9000</value>
#         </property>
#       </configuration>
#
# 5. Configure hdfs-site.xml (C:\hadoop\etc\hadoop\hdfs-site.xml):
#       <configuration>
#         <property>
#           <name>dfs.replication</name>
#           <value>1</value>
#         </property>
#         <property>
#           <name>dfs.namenode.name.dir</name>
#           <value>file:///C:/hadoop/data/namenode</value>
#         </property>
#         <property>
#           <name>dfs.datanode.data.dir</name>
#           <value>file:///C:/hadoop/data/datanode</value>
#         </property>
#       </configuration>
#
# 6. Format the NameNode (run ONCE, first time only):
#       hdfs namenode -format
#
# 7. Start HDFS:
#       start-dfs.cmd
#    Or on Git Bash / WSL:
#       start-dfs.sh
#
# ============================================================================

# --- Configuration ---
HDFS_BASE_DIR="/user/aqi_project/data"
LOCAL_CSV_PATH="../data/india_air_quality.csv"

echo "============================================"
echo " AQI Project — HDFS Data Upload"
echo "============================================"

# Step 1: Create HDFS directory
echo "[1/3] Creating HDFS directory: ${HDFS_BASE_DIR}"
hdfs dfs -mkdir -p ${HDFS_BASE_DIR}

# Step 2: Upload dataset to HDFS
echo "[2/3] Uploading dataset to HDFS..."
hdfs dfs -put -f ${LOCAL_CSV_PATH} ${HDFS_BASE_DIR}/india_air_quality.csv

# Step 3: Verify upload
echo "[3/3] Verifying file on HDFS..."
hdfs dfs -ls ${HDFS_BASE_DIR}/

echo ""
echo "✅ Upload complete!"
echo "   HDFS path: hdfs://localhost:9000${HDFS_BASE_DIR}/india_air_quality.csv"
echo "============================================"
