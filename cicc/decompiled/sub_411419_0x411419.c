// Function: sub_411419
// Address: 0x411419
//
__int64 __fastcall sub_411419(unsigned int *a1)
{
  int v2; // edx
  int v3; // ecx
  int v4; // r8d
  int v5; // r9d
  int v6; // edx
  int v7; // ecx
  int v8; // r8d
  int v9; // r9d
  int v10; // edx
  int v11; // ecx
  int v12; // r8d
  int v13; // edx
  int v14; // ecx
  int v15; // r9d
  int v16; // edx
  int v17; // r8d
  int v18; // r9d
  int v19; // edx
  int v20; // ecx
  int v21; // r8d
  int v22; // r9d
  int v23; // ecx
  int v24; // r8d
  int v25; // r9d
  int v26; // edx
  int v27; // ecx
  int v28; // r8d
  int v29; // r9d
  int v30; // edx
  int v31; // ecx
  int v32; // edx
  int v33; // ecx
  int v34; // r8d
  int v35; // r9d
  int v36; // r8d
  int v37; // r9d
  int v38; // edx
  int v39; // ecx
  int v40; // r8d
  int v41; // r9d
  int v42; // edx
  int v43; // ecx
  int v44; // r8d
  int v45; // r9d
  char *v46; // rsi
  int v47; // edx
  int v48; // ecx
  int v49; // r8d
  int v50; // r9d
  int v51; // edx
  int v52; // ecx
  int v53; // r8d
  int v54; // r9d
  int v55; // edx
  int v56; // ecx
  __int64 v57; // r8
  int v58; // r9d
  int v59; // edx
  int v60; // ecx
  __int64 v61; // r8
  int v62; // r9d
  int v63; // edx
  int v64; // ecx
  int v65; // r8d
  int v66; // r9d
  int v67; // edx
  int v68; // ecx
  int v69; // r8d
  int v70; // edx
  int v71; // ecx
  int v72; // r8d
  int v73; // r9d
  int v74; // ecx
  int v75; // r8d
  int v76; // r9d
  __int64 v77; // rdx
  unsigned __int64 v78; // rdi
  char *v79; // rsi
  int v80; // eax
  int v81; // edx
  int v82; // r8d
  int v83; // r9d
  __int64 *v84; // rcx
  bool v85; // cc
  unsigned __int64 v86; // rdi
  __int64 v87; // r8
  unsigned __int64 v88; // rdi
  __int64 v89; // r8
  unsigned __int64 v90; // rdi
  __int64 v91; // r8
  unsigned __int64 v92; // rdi
  __int64 v93; // r8
  int v94; // edx
  int v95; // ecx
  int v96; // r8d
  int v97; // r9d
  int v98; // edx
  int v99; // ecx
  int v100; // r8d
  int v101; // r9d
  char *v102; // rsi
  int v103; // r8d
  int v104; // r9d
  int v105; // edx
  int v106; // ecx
  unsigned __int64 v107; // rdi
  int v108; // edx
  int v109; // ecx
  int v110; // r8d
  int v111; // r9d
  unsigned int v112; // r13d
  unsigned __int64 v113; // rdi
  __int64 v114; // r8
  int v115; // edx
  int v116; // ecx
  int v117; // r8d
  int v118; // r9d
  int v120; // [rsp-10h] [rbp-1C0h]
  int v121; // [rsp-10h] [rbp-1C0h]
  int v122; // [rsp-10h] [rbp-1C0h]
  int v123; // [rsp-10h] [rbp-1C0h]
  char v124; // [rsp-10h] [rbp-1C0h]
  int v125; // [rsp-10h] [rbp-1C0h]
  __int64 v126; // [rsp-10h] [rbp-1C0h]
  int v127; // [rsp-8h] [rbp-1B8h]
  int v128; // [rsp-8h] [rbp-1B8h]
  int v129; // [rsp-8h] [rbp-1B8h]
  int v130; // [rsp-8h] [rbp-1B8h]
  int v131; // [rsp-8h] [rbp-1B8h]
  int v132; // [rsp-8h] [rbp-1B8h]
  char v133; // [rsp+0h] [rbp-1B0h]
  char v134; // [rsp+0h] [rbp-1B0h]
  unsigned int v135; // [rsp+28h] [rbp-188h]
  char v136; // [rsp+3Ah] [rbp-176h] BYREF
  char v137; // [rsp+3Bh] [rbp-175h] BYREF
  const char *v138; // [rsp+3Ch] [rbp-174h] BYREF
  const char *v139; // [rsp+44h] [rbp-16Ch] BYREF
  const char *v140; // [rsp+4Ch] [rbp-164h] BYREF
  const char *v141; // [rsp+58h] [rbp-158h] BYREF
  const char *v142; // [rsp+60h] [rbp-150h] BYREF
  const char *v143; // [rsp+68h] [rbp-148h] BYREF
  const char *v144; // [rsp+70h] [rbp-140h] BYREF
  const char *v145; // [rsp+78h] [rbp-138h] BYREF
  __int64 v146; // [rsp+80h] [rbp-130h] BYREF
  __int64 v147; // [rsp+88h] [rbp-128h] BYREF
  __int64 v148; // [rsp+90h] [rbp-120h] BYREF
  __int64 v149; // [rsp+98h] [rbp-118h] BYREF
  __int64 v150; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v151; // [rsp+A8h] [rbp-108h] BYREF
  __int64 v152; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v153; // [rsp+B8h] [rbp-F8h] BYREF
  __int64 v154; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v155; // [rsp+C8h] [rbp-E8h] BYREF
  __int64 v156; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v157; // [rsp+D8h] [rbp-D8h] BYREF
  __int64 v158; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v159; // [rsp+E8h] [rbp-C8h] BYREF
  __int64 v160; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v161; // [rsp+F8h] [rbp-B8h] BYREF
  __int64 v162; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v163; // [rsp+108h] [rbp-A8h] BYREF
  const char *v164[7]; // [rsp+110h] [rbp-A0h] BYREF
  const char *v165[13]; // [rsp+148h] [rbp-68h] BYREF

  v165[0] = (const char *)8;
  v146 = 1;
  v147 = 4;
  v151 = 8;
  v152 = 8;
  v153 = 8;
  v148 = 4;
  v150 = 8;
  v149 = 8;
  if ( (unsigned int)sub_1308610("version", (char *)&v140 + 4, v165, 0, 0) )
  {
    sub_130ACF0((unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n", (unsigned int)"version", v2, v3, v4, v5);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"version", (int)"Version", 8, (const char **)((char *)&v140 + 4), 0, 0, 0);
  sub_40ED43(a1, (__int64)"config", (int)"Build-time option settings");
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.cache_oblivious", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.cache_oblivious",
      v6,
      v7,
      v8,
      v9);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"cache_oblivious", (int)"config.cache_oblivious", 0, (const char **)&v136, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.debug", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.debug",
      v10,
      v11,
      v12,
      v120);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"debug", (int)"config.debug", 0, (const char **)&v136, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.fill", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.fill",
      v13,
      v14,
      v127,
      v15);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"fill", (int)"config.fill", 0, (const char **)&v136, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.lazy_lock", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.lazy_lock",
      v16,
      v121,
      v17,
      v18);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"lazy_lock", (int)"config.lazy_lock", 0, (const char **)&v136, 0, 0, 0);
  sub_411329((__int64)a1, (__int64)"malloc_conf", (int)"config.malloc_conf", 8, (const char **)&off_497FB58, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.opt_safety_checks", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.opt_safety_checks",
      v19,
      v20,
      v21,
      v22);
    abort();
  }
  sub_411329(
    (__int64)a1,
    (__int64)"opt_safety_checks",
    (int)"config.opt_safety_checks",
    0,
    (const char **)&v136,
    0,
    0,
    0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.prof", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.prof",
      v128,
      v23,
      v24,
      v25);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"prof", (int)"config.prof", 0, (const char **)&v136, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.prof_libgcc", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.prof_libgcc",
      v26,
      v27,
      v28,
      v29);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"prof_libgcc", (int)"config.prof_libgcc", 0, (const char **)&v136, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.prof_libunwind", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.prof_libunwind",
      v30,
      v31,
      v122,
      v129);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"prof_libunwind", (int)"config.prof_libunwind", 0, (const char **)&v136, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.stats", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.stats",
      v32,
      v33,
      v34,
      v35);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"stats", (int)"config.stats", 0, (const char **)&v136, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.utrace", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.utrace",
      v123,
      v130,
      v36,
      v37);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"utrace", (int)"config.utrace", 0, (const char **)&v136, 0, 0, 0);
  v165[0] = (const char *)1;
  if ( (unsigned int)sub_1308610("config.xmalloc", &v136, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"config.xmalloc",
      v38,
      v39,
      v40,
      v41);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"xmalloc", (int)"config.xmalloc", 0, (const char **)&v136, 0, 0, 0);
  sub_40EB40(a1, (__int64)"xmalloc", v42, v43, v44, v45, v124);
  sub_40ED43(a1, (__int64)"opt", (int)"Run-time option settings");
  if ( !(unsigned int)sub_1308610("opt.abort", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"abort", (int)"opt.abort", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.abort_conf", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"abort_conf", (int)"opt.abort_conf", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.cache_oblivious", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"cache_oblivious", (int)"opt.cache_oblivious", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.confirm_conf", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"confirm_conf", (int)"opt.confirm_conf", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.retain", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"retain", (int)"opt.retain", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.dss", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329((__int64)a1, (__int64)"dss", (int)"opt.dss", 8, (const char **)((char *)&v140 + 4), 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.narenas", &v138, &v147, 0, 0) )
    sub_411329((__int64)a1, (__int64)"narenas", (int)"opt.narenas", 3, &v138, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.percpu_arena", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"percpu_arena",
      (int)"opt.percpu_arena",
      8,
      (const char **)((char *)&v140 + 4),
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.oversize_threshold", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"oversize_threshold", (int)"opt.oversize_threshold", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.hpa", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"hpa", (int)"opt.hpa", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.hpa_slab_max_alloc", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"hpa_slab_max_alloc", (int)"opt.hpa_slab_max_alloc", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.hpa_hugification_threshold", &v145, &v151, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"hpa_hugification_threshold",
      (int)"opt.hpa_hugification_threshold",
      6,
      &v145,
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.hpa_hugify_delay_ms", &v141, &v149, 0, 0) )
    sub_411329((__int64)a1, (__int64)"hpa_hugify_delay_ms", (int)"opt.hpa_hugify_delay_ms", 5, &v141, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.hpa_min_purge_interval_ms", &v141, &v149, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"hpa_min_purge_interval_ms",
      (int)"opt.hpa_min_purge_interval_ms",
      5,
      &v141,
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.hpa_dirty_mult", (char *)&v138 + 4, &v148, 0, 0) )
  {
    if ( HIDWORD(v138) == -1 )
    {
      v165[0] = "-1";
      sub_411329((__int64)a1, (__int64)"hpa_dirty_mult", (int)"opt.hpa_dirty_mult", 8, v165, 0, 0, 0);
    }
    else
    {
      sub_1346A60(HIDWORD(v138), v165);
      v164[0] = (const char *)v165;
      sub_411329((__int64)a1, (__int64)"hpa_dirty_mult", (int)"opt.hpa_dirty_mult", 8, v164, 0, 0, 0);
    }
  }
  if ( !(unsigned int)sub_1308610("opt.hpa_sec_nshards", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"hpa_sec_nshards", (int)"opt.hpa_sec_nshards", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.hpa_sec_max_alloc", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"hpa_sec_max_alloc", (int)"opt.hpa_sec_max_alloc", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.hpa_sec_max_bytes", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"hpa_sec_max_bytes", (int)"opt.hpa_sec_max_bytes", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.hpa_sec_bytes_after_flush", &v145, &v151, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"hpa_sec_bytes_after_flush",
      (int)"opt.hpa_sec_bytes_after_flush",
      6,
      &v145,
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.hpa_sec_batch_fill_extra", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"hpa_sec_batch_fill_extra", (int)"opt.hpa_sec_batch_fill_extra", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.metadata_thp", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"metadata_thp",
      (int)"opt.metadata_thp",
      8,
      (const char **)((char *)&v140 + 4),
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.mutex_max_spin", &v142, &v150, 0, 0) )
    sub_411329((__int64)a1, (__int64)"mutex_max_spin", (int)"opt.mutex_max_spin", 2, &v142, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.background_thread", &v136, &v146, 0, 0)
    && !(unsigned int)sub_1308610("background_thread", &v137, &v146, 0, 0) )
  {
    sub_411329(
      (__int64)a1,
      (__int64)"background_thread",
      (int)"opt.background_thread",
      0,
      (const char **)&v136,
      (__int64)"background_thread",
      0,
      (const char **)&v137);
  }
  if ( !(unsigned int)sub_1308610("opt.dirty_decay_ms", &v143, &v152, 0, 0)
    && !(unsigned int)sub_1308610("arenas.dirty_decay_ms", &v144, &v152, 0, 0) )
  {
    sub_411329(
      (__int64)a1,
      (__int64)"dirty_decay_ms",
      (int)"opt.dirty_decay_ms",
      7,
      &v143,
      (__int64)"arenas.dirty_decay_ms",
      7,
      &v144);
  }
  if ( !(unsigned int)sub_1308610("opt.muzzy_decay_ms", &v143, &v152, 0, 0)
    && !(unsigned int)sub_1308610("arenas.muzzy_decay_ms", &v144, &v152, 0, 0) )
  {
    sub_411329(
      (__int64)a1,
      (__int64)"muzzy_decay_ms",
      (int)"opt.muzzy_decay_ms",
      7,
      &v143,
      (__int64)"arenas.muzzy_decay_ms",
      7,
      &v144);
  }
  if ( !(unsigned int)sub_1308610("opt.lg_extent_max_active_fit", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"lg_extent_max_active_fit", (int)"opt.lg_extent_max_active_fit", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.junk", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329((__int64)a1, (__int64)"junk", (int)"opt.junk", 8, (const char **)((char *)&v140 + 4), 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.zero", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"zero", (int)"opt.zero", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.utrace", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"utrace", (int)"opt.utrace", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.xmalloc", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"xmalloc", (int)"opt.xmalloc", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.experimental_infallible_new", &v136, &v146, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"experimental_infallible_new",
      (int)"opt.experimental_infallible_new",
      0,
      (const char **)&v136,
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.tcache", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"tcache", (int)"opt.tcache", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.tcache_max", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"tcache_max", (int)"opt.tcache_max", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.tcache_nslots_small_min", &v138, &v147, 0, 0) )
    sub_411329((__int64)a1, (__int64)"tcache_nslots_small_min", (int)"opt.tcache_nslots_small_min", 3, &v138, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.tcache_nslots_small_max", &v138, &v147, 0, 0) )
    sub_411329((__int64)a1, (__int64)"tcache_nslots_small_max", (int)"opt.tcache_nslots_small_max", 3, &v138, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.tcache_nslots_large", &v138, &v147, 0, 0) )
    sub_411329((__int64)a1, (__int64)"tcache_nslots_large", (int)"opt.tcache_nslots_large", 3, &v138, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.lg_tcache_nslots_mul", &v143, &v152, 0, 0) )
    sub_411329((__int64)a1, (__int64)"lg_tcache_nslots_mul", (int)"opt.lg_tcache_nslots_mul", 7, &v143, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.tcache_gc_incr_bytes", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"tcache_gc_incr_bytes", (int)"opt.tcache_gc_incr_bytes", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.tcache_gc_delay_bytes", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"tcache_gc_delay_bytes", (int)"opt.tcache_gc_delay_bytes", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.lg_tcache_flush_small_div", &v138, &v147, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"lg_tcache_flush_small_div",
      (int)"opt.lg_tcache_flush_small_div",
      3,
      &v138,
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.lg_tcache_flush_large_div", &v138, &v147, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"lg_tcache_flush_large_div",
      (int)"opt.lg_tcache_flush_large_div",
      3,
      &v138,
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.debug_double_free_max_scan", &v138, &v147, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"debug_double_free_max_scan",
      (int)"opt.debug_double_free_max_scan",
      3,
      &v138,
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.thp", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329((__int64)a1, (__int64)"thp", (int)"opt.thp", 8, (const char **)((char *)&v140 + 4), 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.prof", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"prof", (int)"opt.prof", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.prof_bt_max", &v138, &v147, 0, 0) )
    sub_411329((__int64)a1, (__int64)"prof_bt_max", (int)"opt.prof_bt_max", 3, &v138, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.prof_prefix", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"prof_prefix",
      (int)"opt.prof_prefix",
      8,
      (const char **)((char *)&v140 + 4),
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.prof_active", &v136, &v146, 0, 0)
    && !(unsigned int)sub_1308610("prof.active", &v137, &v146, 0, 0) )
  {
    sub_411329(
      (__int64)a1,
      (__int64)"prof_active",
      (int)"opt.prof_active",
      0,
      (const char **)&v136,
      (__int64)"prof.active",
      0,
      (const char **)&v137);
  }
  if ( !(unsigned int)sub_1308610("opt.prof_thread_active_init", &v136, &v146, 0, 0)
    && !(unsigned int)sub_1308610("prof.thread_active_init", &v137, &v146, 0, 0) )
  {
    sub_411329(
      (__int64)a1,
      (__int64)"prof_thread_active_init",
      (int)"opt.prof_thread_active_init",
      0,
      (const char **)&v136,
      (__int64)"prof.thread_active_init",
      0,
      (const char **)&v137);
  }
  if ( !(unsigned int)sub_1308610("opt.lg_prof_sample", &v143, &v152, 0, 0)
    && !(unsigned int)sub_1308610("prof.lg_sample", &v144, &v152, 0, 0) )
  {
    sub_411329(
      (__int64)a1,
      (__int64)"lg_prof_sample",
      (int)"opt.lg_prof_sample",
      7,
      &v143,
      (__int64)"prof.lg_sample",
      7,
      &v144);
  }
  if ( !(unsigned int)sub_1308610("opt.prof_accum", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"prof_accum", (int)"opt.prof_accum", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.lg_prof_interval", &v143, &v152, 0, 0) )
    sub_411329((__int64)a1, (__int64)"lg_prof_interval", (int)"opt.lg_prof_interval", 7, &v143, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.prof_gdump", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"prof_gdump", (int)"opt.prof_gdump", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.prof_final", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"prof_final", (int)"opt.prof_final", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.prof_leak", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"prof_leak", (int)"opt.prof_leak", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.prof_leak_error", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"prof_leak_error", (int)"opt.prof_leak_error", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.stats_print", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"stats_print", (int)"opt.stats_print", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.stats_print_opts", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"stats_print_opts",
      (int)"opt.stats_print_opts",
      8,
      (const char **)((char *)&v140 + 4),
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.stats_print", &v136, &v146, 0, 0) )
    sub_411329((__int64)a1, (__int64)"stats_print", (int)"opt.stats_print", 0, (const char **)&v136, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.stats_print_opts", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"stats_print_opts",
      (int)"opt.stats_print_opts",
      8,
      (const char **)((char *)&v140 + 4),
      0,
      0,
      0);
  if ( !(unsigned int)sub_1308610("opt.stats_interval", &v142, &v150, 0, 0) )
    sub_411329((__int64)a1, (__int64)"stats_interval", (int)"opt.stats_interval", 2, &v142, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("opt.stats_interval_opts", (char *)&v140 + 4, &v153, 0, 0) )
    sub_411329(
      (__int64)a1,
      (__int64)"stats_interval_opts",
      (int)"opt.stats_interval_opts",
      8,
      (const char **)((char *)&v140 + 4),
      0,
      0,
      0);
  v46 = (char *)&v140 + 4;
  if ( !(unsigned int)sub_1308610("opt.zero_realloc", (char *)&v140 + 4, &v153, 0, 0) )
  {
    v46 = "zero_realloc";
    sub_411329(
      (__int64)a1,
      (__int64)"zero_realloc",
      (int)"opt.zero_realloc",
      8,
      (const char **)((char *)&v140 + 4),
      0,
      0,
      0);
  }
  sub_40EB40(a1, (__int64)v46, v47, v48, v49, v50, v133);
  sub_130F560(a1, "arenas");
  v165[0] = (const char *)4;
  if ( (unsigned int)sub_1308610("arenas.narenas", &v138, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.narenas",
      v51,
      v52,
      v53,
      v54);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"narenas", (int)"Arenas", 3, &v138, 0, 0, 0);
  v165[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("arenas.dirty_decay_ms", &v143, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.dirty_decay_ms",
      v55,
      v56,
      v57,
      v58);
    abort();
  }
  sub_40EDDD((__int64)a1, (__int64)"dirty_decay_ms", 7, &v143, v57);
  v165[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("arenas.muzzy_decay_ms", &v143, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.muzzy_decay_ms",
      v59,
      v60,
      v61,
      v62);
    abort();
  }
  sub_40EDDD((__int64)a1, (__int64)"muzzy_decay_ms", 7, &v143, v61);
  v165[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("arenas.quantum", &v145, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.quantum",
      v63,
      v64,
      v65,
      v66);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"quantum", (int)"Quantum size", 6, &v145, 0, 0, 0);
  v165[0] = (const char *)8;
  if ( (unsigned int)sub_1308610("arenas.page", &v145, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.page",
      v67,
      v68,
      v69,
      v125);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"page", (int)"Page size", 6, &v145, 0, 0, 0);
  if ( !(unsigned int)sub_1308610("arenas.tcache_max", &v145, &v151, 0, 0) )
    sub_411329((__int64)a1, (__int64)"tcache_max", (int)"Maximum thread-cached size class", 6, &v145, 0, 0, 0);
  v165[0] = (const char *)4;
  if ( (unsigned int)sub_1308610("arenas.nbins", &v139, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.nbins",
      v70,
      v71,
      v72,
      v73);
    abort();
  }
  sub_411329((__int64)a1, (__int64)"nbins", (int)"Number of bin size classes", 3, &v139, 0, 0, 0);
  v165[0] = (const char *)4;
  if ( (unsigned int)sub_1308610("arenas.nhbins", (char *)&v139 + 4, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.nhbins",
      v131,
      v74,
      v75,
      v76);
    abort();
  }
  sub_411329(
    (__int64)a1,
    (__int64)"nhbins",
    (int)"Number of thread-cache bin size classes",
    3,
    (const char **)((char *)&v139 + 4),
    0,
    0,
    0);
  if ( *a1 <= 1 )
  {
    sub_40EDA0((__int64)a1, (__int64)"bin", v77);
    v162 = 7;
    v78 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      v78 = sub_1313D30(v78, 0);
    v79 = (char *)v164;
    v80 = sub_133D570(v78, v164, 0, "arenas.bin", &v162);
    v84 = &v154;
    v135 = 0;
    if ( v80 )
    {
LABEL_177:
      sub_130AA40("<jemalloc>: Failure in ctl_mibnametomib()\n");
      abort();
    }
    while ( v135 < (unsigned int)v139 )
    {
      v85 = *a1 <= 1;
      v164[2] = (const char *)v135;
      if ( v85 )
        sub_130F360(a1);
      v154 = 7;
      v155 = 8;
      v86 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v86) = sub_1313D30(v86, 0);
      if ( (unsigned int)sub_133D620(
                           v86,
                           (unsigned int)v164,
                           3,
                           (unsigned int)"size",
                           (unsigned int)&v154,
                           (unsigned int)&v145,
                           (__int64)&v155,
                           0,
                           0) )
        goto LABEL_184;
      sub_40EDDD((__int64)a1, (__int64)"size", 6, &v145, v87);
      v156 = 7;
      v157 = 4;
      v88 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v88) = sub_1313D30(v88, 0);
      if ( (unsigned int)sub_133D620(
                           v88,
                           (unsigned int)v164,
                           3,
                           (unsigned int)"nregs",
                           (unsigned int)&v156,
                           (unsigned int)&v138 + 4,
                           (__int64)&v157,
                           0,
                           0) )
        goto LABEL_184;
      sub_40EDDD((__int64)a1, (__int64)"nregs", 4, (const char **)((char *)&v138 + 4), v89);
      v158 = 7;
      v159 = 8;
      v90 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v90) = sub_1313D30(v90, 0);
      if ( (unsigned int)sub_133D620(
                           v90,
                           (unsigned int)v164,
                           3,
                           (unsigned int)"slab_size",
                           (unsigned int)&v158,
                           (unsigned int)&v145,
                           (__int64)&v159,
                           0,
                           0) )
        goto LABEL_184;
      sub_40EDDD((__int64)a1, (__int64)"slab_size", 6, &v145, v91);
      v160 = 7;
      v161 = 4;
      v92 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v92) = sub_1313D30(v92, 0);
      if ( (unsigned int)sub_133D620(
                           v92,
                           (unsigned int)v164,
                           3,
                           (unsigned int)"nshards",
                           (unsigned int)&v160,
                           (unsigned int)&v138 + 4,
                           (__int64)&v161,
                           0,
                           0) )
      {
LABEL_184:
        sub_130AA40("<jemalloc>: Failure in ctl_bymibname()\n");
        abort();
      }
      v79 = "nshards";
      sub_40EDDD((__int64)a1, (__int64)"nshards", 4, (const char **)((char *)&v138 + 4), v93);
      sub_40E56D(a1, (__int64)"nshards", v94, v95, v96, v97, v134);
      ++v135;
    }
    sub_40E525(a1, (__int64)v79, v81, (int)v84, v82, v83, v134);
  }
  v165[0] = (const char *)4;
  if ( (unsigned int)sub_1308610("arenas.nlextents", &v140, v165, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.nlextents",
      v98,
      v99,
      v100,
      v101);
    abort();
  }
  v102 = "nlextents";
  sub_411329((__int64)a1, (__int64)"nlextents", (int)"Number of large size classes", 3, &v140, 0, 0, 0);
  v105 = v126;
  v106 = v132;
  if ( *a1 <= 1 )
  {
    sub_40EDA0((__int64)a1, (__int64)"lextent", v126);
    v163 = 7;
    v107 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      v107 = sub_1313D30(v107, 0);
    v102 = (char *)v165;
    if ( (unsigned int)sub_133D570(v107, v165, 0, "arenas.lextent", &v163) )
      goto LABEL_177;
    v112 = 0;
    while ( v112 < (unsigned int)v140 )
    {
      v85 = *a1 <= 1;
      v165[2] = (const char *)v112;
      if ( v85 )
        sub_130F360(a1);
      v162 = 7;
      v163 = 8;
      v113 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v113) = sub_1313D30(v113, 0);
      if ( (unsigned int)sub_133D620(
                           v113,
                           (unsigned int)v165,
                           3,
                           (unsigned int)"size",
                           (unsigned int)&v162,
                           (unsigned int)&v145,
                           (__int64)&v163,
                           0,
                           0) )
        goto LABEL_184;
      ++v112;
      v102 = "size";
      sub_40EDDD((__int64)a1, (__int64)"size", 6, &v145, v114);
      sub_40E56D(a1, (__int64)"size", v115, v116, v117, v118, v134);
    }
    sub_40E525(a1, (__int64)v102, v108, v109, v110, v111, v134);
  }
  return sub_40E56D(a1, (__int64)v102, v105, v106, v103, v104, v134);
}
