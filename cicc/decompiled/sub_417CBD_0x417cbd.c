// Function: sub_417CBD
// Address: 0x417cbd
//
char __fastcall sub_417CBD(unsigned int *a1, int a2, int a3, int a4, int a5, char a6, int a7, int a8, int a9)
{
  int v11; // ebx
  int v12; // edx
  int v13; // ecx
  int v14; // r8d
  int v15; // r9d
  int v16; // edx
  int v17; // ecx
  int v18; // r8d
  int v19; // r9d
  int v20; // edx
  int v21; // ecx
  int v22; // r8d
  int v23; // r9d
  int v24; // edx
  int v25; // ecx
  int v26; // r8d
  int v27; // r9d
  int v28; // edx
  int v29; // ecx
  int v30; // r8d
  int v31; // r9d
  int v32; // edx
  int v33; // ecx
  int v34; // r8d
  int v35; // r9d
  int v36; // edx
  int v37; // ecx
  int v38; // r8d
  int v39; // r9d
  int v40; // edx
  int v41; // ecx
  int v42; // r8d
  int v43; // r9d
  int v44; // edx
  int v45; // ecx
  int v46; // r8d
  int v47; // r9d
  int v48; // edx
  int v49; // ecx
  int v50; // r8d
  int v51; // r9d
  int v52; // edx
  int v53; // ecx
  int v54; // r8d
  int v55; // r9d
  __int64 v56; // r8
  __int64 v57; // r8
  __int64 v58; // r8
  __int64 v59; // r8
  __int64 v60; // r8
  __int64 v61; // r8
  __int64 v62; // r8
  __int64 v63; // r8
  int v64; // ecx
  int v65; // r8d
  int v66; // r9d
  __int64 v67; // r8
  __int64 v68; // r8
  __int64 v69; // r8
  int v70; // edx
  int v71; // ecx
  int v72; // r8d
  int v73; // r9d
  __int64 *v74; // rsi
  int v75; // r9d
  int v76; // edx
  int v77; // ecx
  int v78; // r8d
  __int64 v79; // r9
  __int64 v80; // rdx
  int v81; // ecx
  int v82; // r8d
  int v83; // r9d
  int v84; // edx
  int v85; // ecx
  int v86; // r8d
  int v87; // r9d
  unsigned __int64 v88; // rdi
  __int64 *v89; // r14
  void *v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r8
  int v93; // r9d
  int v94; // edx
  int v95; // ecx
  int v96; // r8d
  int v97; // r9d
  int v98; // ecx
  int v99; // r8d
  int v100; // r9d
  char result; // al
  int v102; // edx
  int v103; // ecx
  int v104; // r8d
  int v105; // r9d
  unsigned int v106; // r13d
  void *v107; // rsp
  int v108; // eax
  int v109; // edx
  int v110; // r8d
  int v111; // r9d
  __int64 *v112; // rcx
  unsigned int i; // r14d
  __int64 v114; // rsi
  char v115; // bl
  int v116; // edx
  int v117; // ecx
  const char *v118; // r8
  int v119; // r9d
  int v120; // edx
  int v121; // ecx
  int v122; // r8d
  int v123; // r9d
  int v124; // edx
  int v125; // ecx
  int v126; // r8d
  int v127; // r9d
  char v128; // r15
  unsigned int v129; // ebx
  char v130; // r14
  int v131; // ecx
  int v132; // r8d
  int v133; // r9d
  int v134; // edx
  int v135; // ecx
  int v136; // r8d
  int v137; // r9d
  char v138; // [rsp-10h] [rbp-370h]
  int v139; // [rsp-10h] [rbp-370h]
  char v140; // [rsp-10h] [rbp-370h]
  char v141; // [rsp-10h] [rbp-370h]
  char v142; // [rsp-10h] [rbp-370h]
  int v143; // [rsp-10h] [rbp-370h]
  int v144; // [rsp-8h] [rbp-368h]
  int v145; // [rsp-8h] [rbp-368h]
  __int64 *v146; // [rsp+0h] [rbp-360h] BYREF
  int v147; // [rsp+Ch] [rbp-354h]
  char v148; // [rsp+13h] [rbp-34Dh]
  char v149; // [rsp+14h] [rbp-34Ch]
  char v150; // [rsp+15h] [rbp-34Bh]
  unsigned __int8 v151; // [rsp+16h] [rbp-34Ah]
  unsigned __int8 v152; // [rsp+17h] [rbp-349h]
  __int64 **v153; // [rsp+18h] [rbp-348h]
  int v154; // [rsp+20h] [rbp-340h]
  int v155; // [rsp+24h] [rbp-33Ch]
  int v156; // [rsp+28h] [rbp-338h]
  int v157; // [rsp+2Ch] [rbp-334h]
  int v158; // [rsp+30h] [rbp-330h]
  int v159; // [rsp+34h] [rbp-32Ch]
  __int64 *v160; // [rsp+38h] [rbp-328h]
  char v161; // [rsp+4Bh] [rbp-315h] BYREF
  unsigned int v162; // [rsp+4Ch] [rbp-314h] BYREF
  const char *v163; // [rsp+50h] [rbp-310h] BYREF
  const char *v164; // [rsp+58h] [rbp-308h] BYREF
  const char *v165; // [rsp+60h] [rbp-300h] BYREF
  const char *v166; // [rsp+68h] [rbp-2F8h] BYREF
  const char *v167; // [rsp+70h] [rbp-2F0h] BYREF
  const char *v168; // [rsp+78h] [rbp-2E8h] BYREF
  const char *v169; // [rsp+80h] [rbp-2E0h] BYREF
  const char *v170; // [rsp+88h] [rbp-2D8h] BYREF
  const char *v171; // [rsp+90h] [rbp-2D0h] BYREF
  const char *v172; // [rsp+98h] [rbp-2C8h] BYREF
  const char *v173; // [rsp+A0h] [rbp-2C0h] BYREF
  __int64 v174; // [rsp+A8h] [rbp-2B8h] BYREF
  unsigned __int64 v175; // [rsp+B0h] [rbp-2B0h] BYREF
  __int64 v176; // [rsp+B8h] [rbp-2A8h] BYREF
  __int64 v177; // [rsp+C0h] [rbp-2A0h] BYREF
  __int64 v178; // [rsp+C8h] [rbp-298h] BYREF
  __int64 v179; // [rsp+D0h] [rbp-290h] BYREF
  __int64 v180; // [rsp+D8h] [rbp-288h] BYREF
  __int64 v181; // [rsp+E0h] [rbp-280h]
  _BYTE v182[40]; // [rsp+F0h] [rbp-270h] BYREF
  _BYTE v183[40]; // [rsp+118h] [rbp-248h] BYREF
  _BYTE v184[16]; // [rsp+140h] [rbp-220h] BYREF
  __int64 v185; // [rsp+150h] [rbp-210h]
  const char *v186[61]; // [rsp+178h] [rbp-1E8h] BYREF

  v11 = a7;
  v158 = a5;
  v155 = a2;
  v157 = a8;
  v154 = a3;
  v156 = a9;
  v159 = a4;
  v152 = a5;
  v151 = a6;
  v149 = a8;
  v150 = a7;
  v148 = a9;
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.allocated", &v163, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.allocated",
      v12,
      v13,
      v14,
      v15);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.active", &v164, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.active",
      v16,
      v17,
      v18,
      v19);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.metadata", &v165, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.metadata",
      v20,
      v21,
      v22,
      v23);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.metadata_thp", &v166, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.metadata_thp",
      v24,
      v25,
      v26,
      v27);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.resident", &v167, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.resident",
      v28,
      v29,
      v30,
      v31);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.mapped", &v168, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.mapped",
      v32,
      v33,
      v34,
      v35);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.retained", &v169, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.retained",
      v36,
      v37,
      v38,
      v39);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.zero_reallocs", &v171, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.zero_reallocs",
      v40,
      v41,
      v42,
      v43);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.background_thread.num_threads", &v170, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.background_thread.num_threads",
      v44,
      v45,
      v46,
      v47);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.background_thread.num_runs", &v172, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.background_thread.num_runs",
      v48,
      v49,
      v50,
      v51);
    abort();
  }
  v186[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("stats.background_thread.run_interval", &v173, v186, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"stats.background_thread.run_interval",
      v52,
      v53,
      v54,
      v55);
    abort();
  }
  sub_130F560(a1, "stats");
  sub_40EDDD((__int64)a1, (__int64)"allocated", 6, &v163, v56);
  sub_40EDDD((__int64)a1, (__int64)"active", 6, &v164, v57);
  sub_40EDDD((__int64)a1, (__int64)"metadata", 6, &v165, v58);
  sub_40EDDD((__int64)a1, (__int64)"metadata_thp", 6, &v166, v59);
  sub_40EDDD((__int64)a1, (__int64)"resident", 6, &v167, v60);
  sub_40EDDD((__int64)a1, (__int64)"mapped", 6, &v168, v61);
  sub_40EDDD((__int64)a1, (__int64)"retained", 6, &v169, v62);
  sub_40EDDD((__int64)a1, (__int64)"zero_reallocs", 6, &v171, v63);
  sub_130F1C0(
    (_DWORD)a1,
    (unsigned int)"Allocated: %zu, active: %zu, metadata: %zu (n_thp %zu), resident: %zu, mapped: %zu, retained: %zu\n",
    (_DWORD)v163,
    (_DWORD)v164,
    (_DWORD)v165,
    (_DWORD)v166);
  sub_130F1C0((_DWORD)a1, (unsigned int)"Count of realloc(non-null-ptr, 0) calls: %zu\n", (_DWORD)v171, v64, v65, v66);
  sub_130F560(a1, "background_thread");
  sub_40EDDD((__int64)a1, (__int64)"num_threads", 6, &v170, v67);
  sub_40EDDD((__int64)a1, (__int64)"num_runs", 5, &v172, v68);
  sub_40EDDD((__int64)a1, (__int64)"run_interval", 5, &v173, v69);
  sub_40E56D(a1, (__int64)"run_interval", v70, v71, v72, v73, (char)v146);
  v74 = (__int64 *)"Background threads: %zu, num_runs: %lu, run_interval: %lu ns\n";
  sub_130F1C0(
    (_DWORD)a1,
    (unsigned int)"Background threads: %zu, num_runs: %lu, run_interval: %lu ns\n",
    (_DWORD)v170,
    (_DWORD)v172,
    (_DWORD)v173,
    v75);
  if ( (_BYTE)a7 )
  {
    v174 = 0;
    sub_40E313((__int64)&v174, (__int64)byte_3F871B3, (__int64)v182, (__int64)v186, (__int64)v183, v79);
    if ( *a1 == 2 )
      sub_40ECF5((int)a1, &v174, v80, v81, v82, v83, (char)v146);
    sub_130F560(a1, "mutexes");
    v176 = 7;
    v177 = 8;
    if ( (unsigned int)sub_13086C0("stats.arenas.0.uptime", v184, &v176) )
    {
      sub_130ACF0(
        (unsigned int)"<jemalloc>: Failure in xmallctlnametomib(\"%s\", ...)\n",
        (unsigned int)"stats.arenas.0.uptime",
        v84,
        v85,
        v86,
        v87);
      abort();
    }
    v185 = 0;
    if ( (unsigned int)sub_1308750(v184, v176, &v175, &v177, 0, 0) )
      goto LABEL_29;
    v180 = 7;
    v88 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      v88 = sub_1313D30(v88, 0);
    if ( (unsigned int)sub_133D570(v88, v184, 0, "stats.mutexes", &v180) )
    {
      sub_130AA40("<jemalloc>: Failure in ctl_mibnametomib()\n");
      abort();
    }
    v89 = (__int64 *)&off_4C6F1A0;
    LODWORD(v153) = a7;
    v90 = &unk_4C6F1E8;
    v160 = (__int64 *)&unk_4C6F1E8;
    do
    {
      v144 = (int)v90;
      v91 = *v89++;
      sub_40E5B5((__int64)v184, 2, v91, (__int64)v182, (__int64)v186, (__int64)v183, v175);
      sub_130F560(a1, *(v89 - 1));
      v74 = &v174;
      sub_40EE2B(a1, &v174, v186, (__int64)v183, v92, v93);
      sub_40E56D(a1, (__int64)&v174, v94, v95, v96, v97, v138);
      LODWORD(v90) = v139;
    }
    while ( v160 != v89 );
    v11 = (int)v153;
    sub_40E56D(a1, (__int64)&v174, v144, v98, v99, v100, (char)v146);
  }
  sub_40E56D(a1, (__int64)v74, v76, v77, v78, v79, (char)v146);
  result = v154 | v159;
  if ( (unsigned __int8)v154 | (unsigned __int8)v159 || (_BYTE)v155 )
  {
    v153 = &v146;
    sub_130F560(a1, "stats.arenas");
    v186[0] = (const char *)4;
    if ( (unsigned int)sub_1308610("arenas.narenas", &v162, v186, 0, 0) )
    {
      sub_130ACF0(
        (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
        (unsigned int)"arenas.narenas",
        v102,
        v103,
        v104,
        v105);
      abort();
    }
    v106 = 0;
    v178 = 3;
    v107 = alloca(16 * (((unsigned __int64)v162 + 15) >> 4));
    v160 = (__int64 *)&v146;
    v108 = sub_13086C0("arena.0.initialized", &v180, &v178);
    v112 = &v179;
    if ( v108 )
    {
      sub_130ACF0(
        (unsigned int)"<jemalloc>: Failure in xmallctlnametomib(\"%s\", ...)\n",
        (unsigned int)"arena.0.initialized",
        v109,
        (unsigned int)&v179,
        v110,
        v111);
      abort();
    }
    v147 = v11;
    for ( i = 0; ; ++i )
    {
      v114 = v178;
      if ( i >= v162 )
        break;
      v181 = i;
      v146 = v112;
      v179 = 1;
      if ( (unsigned int)sub_1308750(&v180, v178, (char *)v160 + i, v112, 0, 0) )
        goto LABEL_29;
      v112 = v146;
      if ( *((_BYTE *)v160 + i) )
        ++v106;
    }
    v181 = 4097;
    v115 = v147;
    v179 = 1;
    if ( (unsigned int)sub_1308750(&v180, v178, &v161, v112, 0, 0) )
    {
LABEL_29:
      sub_130AA40("<jemalloc>: Failure in xmallctlbymib()\n");
      abort();
    }
    if ( (_BYTE)v155 && (v106 > 1 || (_BYTE)v159 != 1) )
    {
      sub_130F1C0((_DWORD)a1, (unsigned int)"Merged arenas stats:\n", v116, v117, (_DWORD)v118, v119);
      sub_130F560(a1, "merged");
      v114 = 4096;
      sub_4134A7(a1, 0x1000u, v158, a6, v115, v157, v156);
      sub_40E56D(a1, 4096, v120, v121, v122, v123, v140);
    }
    if ( v161 && (_BYTE)v154 )
    {
      sub_130F1C0((_DWORD)a1, (unsigned int)"Destroyed arenas stats:\n", v116, v117, (_DWORD)v118, v119);
      sub_130F560(a1, "destroyed");
      v114 = 4097;
      sub_4134A7(a1, 0x1001u, v158, a6, v115, v157, v156);
      sub_40E56D(a1, 4097, v124, v125, v126, v127, v141);
      v118 = "destroyed";
    }
    if ( (_BYTE)v159 )
    {
      v128 = v149;
      v129 = 0;
      v130 = v150;
      v159 = v151;
      v158 = v152;
      while ( v129 < v162 )
      {
        v116 = (int)v160;
        if ( *((_BYTE *)v160 + v129) )
        {
          sub_40E1DF((__int64)v186, 0x14u, (char *)"%u", v129);
          sub_130F560(a1, v186);
          v114 = v129;
          v145 = sub_130F1C0((_DWORD)a1, (unsigned int)"arenas[%s]:\n", (unsigned int)v186, v131, v132, v133);
          sub_4134A7(a1, v129, v158, v159, v130, v128, v148);
          sub_40E56D(a1, v129, v134, v135, v136, v137, v142);
          v116 = v143;
          v117 = v145;
        }
        ++v129;
      }
    }
    return sub_40E56D(a1, v114, v116, v117, (int)v118, v119, (char)v146);
  }
  return result;
}
