// Function: sub_108F690
// Address: 0x108f690
//
__int64 __fastcall sub_108F690(__int64 a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rdi
  __int64 v4; // rsi
  _QWORD *v5; // r14
  _QWORD *v6; // r13
  __int64 v7; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // r15
  _QWORD *v10; // r13
  __int64 v11; // r14
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r10
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 *v22; // r13
  unsigned __int64 v23; // rbx
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // r10
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 *v33; // r13
  unsigned __int64 v34; // rbx
  __int64 v35; // rdi
  __int64 v36; // rsi
  __int64 v37; // r10
  __int64 v38; // r9
  __int64 v39; // r8
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 *v44; // r13
  unsigned __int64 v45; // rbx
  __int64 v46; // rdi
  __int64 v47; // rsi
  __int64 v48; // r10
  __int64 v49; // r9
  __int64 v50; // r8
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 *v55; // r13
  unsigned __int64 v56; // rbx
  __int64 v57; // rdi
  __int64 v58; // rsi
  __int64 v59; // r10
  __int64 v60; // r9
  __int64 v61; // r8
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 *v66; // r13
  unsigned __int64 v67; // rbx
  __int64 v68; // rdi
  __int64 v69; // rsi
  __int64 v70; // r10
  __int64 v71; // r9
  __int64 v72; // r8
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 *v77; // r13
  unsigned __int64 v78; // rbx
  __int64 v79; // rdi
  __int64 v80; // rsi
  __int64 v81; // r10
  __int64 v82; // r9
  __int64 v83; // r8
  __int64 v84; // rcx
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // rdi
  __int64 *v88; // r13
  unsigned __int64 v89; // rbx
  __int64 v90; // rdi
  __int64 v91; // rsi
  __int64 v92; // r10
  __int64 v93; // r9
  __int64 v94; // r8
  __int64 v95; // rcx
  __int64 v96; // rdx
  __int64 v97; // rax
  __int64 v98; // rdi
  __int64 *v99; // r13
  unsigned __int64 v100; // rbx
  __int64 v101; // rdi
  __int64 v102; // rsi
  __int64 v103; // r10
  __int64 v104; // r9
  __int64 v105; // r8
  __int64 v106; // rcx
  __int64 v107; // rdx
  __int64 v108; // rax
  __int64 v109; // rdi
  __int64 *v110; // r13
  unsigned __int64 v111; // rbx
  __int64 v112; // rdi
  __int64 v113; // rsi
  __int64 v114; // rdi
  __int64 v115; // rdi
  __int64 v117; // [rsp+0h] [rbp-70h] BYREF
  __int64 v118; // [rsp+8h] [rbp-68h]
  __int64 v119; // [rsp+10h] [rbp-60h]
  __int64 v120; // [rsp+18h] [rbp-58h]
  __int64 v121; // [rsp+20h] [rbp-50h] BYREF
  __int64 v122; // [rsp+28h] [rbp-48h]
  __int64 v123; // [rsp+30h] [rbp-40h]
  __int64 v124; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 2024);
  *(_QWORD *)a1 = off_49E6238;
  *(_QWORD *)(a1 + 1960) = off_497C0E0;
  if ( v2 )
  {
    v3 = (_QWORD *)v2[4];
    if ( v3 != v2 + 6 )
      j_j___libc_free_0(v3, v2[6] + 1LL);
    if ( (_QWORD *)*v2 != v2 + 2 )
      j_j___libc_free_0(*v2, v2[2] + 1LL);
    j_j___libc_free_0(v2, 72);
  }
  v4 = *(_QWORD *)(a1 + 1920);
  *(_QWORD *)(a1 + 1840) = off_497C0B0;
  sub_108E240(a1 + 1904, (_QWORD *)v4);
  v5 = *(_QWORD **)(a1 + 1824);
  v6 = *(_QWORD **)(a1 + 1816);
  if ( v5 != v6 )
  {
    do
    {
      v7 = *v6;
      v8 = v6;
      v6 += 8;
      (*(void (__fastcall **)(_QWORD *))(v7 + 16))(v8);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD **)(a1 + 1816);
  }
  if ( v6 )
  {
    v4 = *(_QWORD *)(a1 + 1832) - (_QWORD)v6;
    j_j___libc_free_0(v6, v4);
  }
  v9 = *(_QWORD **)(a1 + 1800);
  v10 = *(_QWORD **)(a1 + 1792);
  if ( v9 != v10 )
  {
    do
    {
      v11 = v10[8];
      *v10 = off_497C080;
      if ( v11 )
      {
        v12 = *(_QWORD *)(v11 + 64);
        if ( v12 != v11 + 80 )
          _libc_free(v12, v4);
        v13 = *(_QWORD *)(v11 + 32);
        if ( v13 != v11 + 48 )
          _libc_free(v13, v4);
        v4 = 96;
        j_j___libc_free_0(v11, 96);
      }
      v10 += 10;
    }
    while ( v9 != v10 );
    v10 = *(_QWORD **)(a1 + 1792);
  }
  if ( v10 )
    j_j___libc_free_0(v10, *(_QWORD *)(a1 + 1808) - (_QWORD)v10);
  *(_QWORD *)(a1 + 1608) = off_497C050;
  sub_108AE30((__int64 *)(a1 + 1672));
  *(_QWORD *)(a1 + 1464) = off_497C050;
  sub_108AE30((__int64 *)(a1 + 1528));
  *(_QWORD *)(a1 + 1320) = off_497C050;
  sub_108AE30((__int64 *)(a1 + 1384));
  *(_QWORD *)(a1 + 1176) = off_497C050;
  sub_108AE30((__int64 *)(a1 + 1240));
  *(_QWORD *)(a1 + 1032) = off_497C050;
  sub_108AE30((__int64 *)(a1 + 1096));
  v14 = *(_QWORD *)(a1 + 968);
  v15 = *(_QWORD *)(a1 + 1000);
  v16 = *(_QWORD *)(a1 + 1008);
  v17 = *(_QWORD *)(a1 + 1016);
  v18 = *(_QWORD *)(a1 + 976);
  v124 = *(_QWORD *)(a1 + 1024);
  v19 = *(_QWORD *)(a1 + 984);
  v20 = *(_QWORD *)(a1 + 992);
  v117 = v14;
  v121 = v15;
  v122 = v16;
  v123 = v17;
  v118 = v18;
  v119 = v19;
  v120 = v20;
  sub_108B970(&v117, &v121);
  v21 = *(_QWORD *)(a1 + 952);
  if ( v21 )
  {
    v22 = *(__int64 **)(a1 + 992);
    v23 = *(_QWORD *)(a1 + 1024) + 8LL;
    if ( v23 > (unsigned __int64)v22 )
    {
      do
      {
        v24 = *v22++;
        j_j___libc_free_0(v24, 480);
      }
      while ( v23 > (unsigned __int64)v22 );
      v21 = *(_QWORD *)(a1 + 952);
    }
    j_j___libc_free_0(v21, 8LL * *(_QWORD *)(a1 + 960));
  }
  v25 = *(_QWORD *)(a1 + 888);
  v26 = *(_QWORD *)(a1 + 920);
  v27 = *(_QWORD *)(a1 + 928);
  v28 = *(_QWORD *)(a1 + 936);
  v29 = *(_QWORD *)(a1 + 896);
  v124 = *(_QWORD *)(a1 + 944);
  v30 = *(_QWORD *)(a1 + 904);
  v31 = *(_QWORD *)(a1 + 912);
  v117 = v25;
  v121 = v26;
  v122 = v27;
  v123 = v28;
  v118 = v29;
  v119 = v30;
  v120 = v31;
  sub_108B970(&v117, &v121);
  v32 = *(_QWORD *)(a1 + 872);
  if ( v32 )
  {
    v33 = *(__int64 **)(a1 + 912);
    v34 = *(_QWORD *)(a1 + 944) + 8LL;
    if ( v34 > (unsigned __int64)v33 )
    {
      do
      {
        v35 = *v33++;
        j_j___libc_free_0(v35, 480);
      }
      while ( v34 > (unsigned __int64)v33 );
      v32 = *(_QWORD *)(a1 + 872);
    }
    j_j___libc_free_0(v32, 8LL * *(_QWORD *)(a1 + 880));
  }
  v36 = *(_QWORD *)(a1 + 808);
  v37 = *(_QWORD *)(a1 + 840);
  v38 = *(_QWORD *)(a1 + 848);
  v39 = *(_QWORD *)(a1 + 856);
  v40 = *(_QWORD *)(a1 + 816);
  v124 = *(_QWORD *)(a1 + 864);
  v41 = *(_QWORD *)(a1 + 824);
  v42 = *(_QWORD *)(a1 + 832);
  v117 = v36;
  v121 = v37;
  v122 = v38;
  v123 = v39;
  v118 = v40;
  v119 = v41;
  v120 = v42;
  sub_108B970(&v117, &v121);
  v43 = *(_QWORD *)(a1 + 792);
  if ( v43 )
  {
    v44 = *(__int64 **)(a1 + 832);
    v45 = *(_QWORD *)(a1 + 864) + 8LL;
    if ( v45 > (unsigned __int64)v44 )
    {
      do
      {
        v46 = *v44++;
        j_j___libc_free_0(v46, 480);
      }
      while ( v45 > (unsigned __int64)v44 );
      v43 = *(_QWORD *)(a1 + 792);
    }
    j_j___libc_free_0(v43, 8LL * *(_QWORD *)(a1 + 800));
  }
  v47 = *(_QWORD *)(a1 + 728);
  v48 = *(_QWORD *)(a1 + 760);
  v49 = *(_QWORD *)(a1 + 768);
  v50 = *(_QWORD *)(a1 + 776);
  v51 = *(_QWORD *)(a1 + 736);
  v124 = *(_QWORD *)(a1 + 784);
  v52 = *(_QWORD *)(a1 + 744);
  v53 = *(_QWORD *)(a1 + 752);
  v117 = v47;
  v121 = v48;
  v122 = v49;
  v123 = v50;
  v118 = v51;
  v119 = v52;
  v120 = v53;
  sub_108B970(&v117, &v121);
  v54 = *(_QWORD *)(a1 + 712);
  if ( v54 )
  {
    v55 = *(__int64 **)(a1 + 752);
    v56 = *(_QWORD *)(a1 + 784) + 8LL;
    if ( v56 > (unsigned __int64)v55 )
    {
      do
      {
        v57 = *v55++;
        j_j___libc_free_0(v57, 480);
      }
      while ( v56 > (unsigned __int64)v55 );
      v54 = *(_QWORD *)(a1 + 712);
    }
    j_j___libc_free_0(v54, 8LL * *(_QWORD *)(a1 + 720));
  }
  v58 = *(_QWORD *)(a1 + 648);
  v59 = *(_QWORD *)(a1 + 680);
  v60 = *(_QWORD *)(a1 + 688);
  v61 = *(_QWORD *)(a1 + 696);
  v62 = *(_QWORD *)(a1 + 656);
  v124 = *(_QWORD *)(a1 + 704);
  v63 = *(_QWORD *)(a1 + 664);
  v64 = *(_QWORD *)(a1 + 672);
  v117 = v58;
  v121 = v59;
  v122 = v60;
  v123 = v61;
  v118 = v62;
  v119 = v63;
  v120 = v64;
  sub_108B970(&v117, &v121);
  v65 = *(_QWORD *)(a1 + 632);
  if ( v65 )
  {
    v66 = *(__int64 **)(a1 + 672);
    v67 = *(_QWORD *)(a1 + 704) + 8LL;
    if ( v67 > (unsigned __int64)v66 )
    {
      do
      {
        v68 = *v66++;
        j_j___libc_free_0(v68, 480);
      }
      while ( v67 > (unsigned __int64)v66 );
      v65 = *(_QWORD *)(a1 + 632);
    }
    j_j___libc_free_0(v65, 8LL * *(_QWORD *)(a1 + 640));
  }
  v69 = *(_QWORD *)(a1 + 568);
  v70 = *(_QWORD *)(a1 + 600);
  v71 = *(_QWORD *)(a1 + 608);
  v72 = *(_QWORD *)(a1 + 616);
  v73 = *(_QWORD *)(a1 + 576);
  v124 = *(_QWORD *)(a1 + 624);
  v74 = *(_QWORD *)(a1 + 584);
  v75 = *(_QWORD *)(a1 + 592);
  v117 = v69;
  v121 = v70;
  v122 = v71;
  v123 = v72;
  v118 = v73;
  v119 = v74;
  v120 = v75;
  sub_108B970(&v117, &v121);
  v76 = *(_QWORD *)(a1 + 552);
  if ( v76 )
  {
    v77 = *(__int64 **)(a1 + 592);
    v78 = *(_QWORD *)(a1 + 624) + 8LL;
    if ( v78 > (unsigned __int64)v77 )
    {
      do
      {
        v79 = *v77++;
        j_j___libc_free_0(v79, 480);
      }
      while ( v78 > (unsigned __int64)v77 );
      v76 = *(_QWORD *)(a1 + 552);
    }
    j_j___libc_free_0(v76, 8LL * *(_QWORD *)(a1 + 560));
  }
  v80 = *(_QWORD *)(a1 + 488);
  v81 = *(_QWORD *)(a1 + 520);
  v82 = *(_QWORD *)(a1 + 528);
  v83 = *(_QWORD *)(a1 + 536);
  v84 = *(_QWORD *)(a1 + 496);
  v124 = *(_QWORD *)(a1 + 544);
  v85 = *(_QWORD *)(a1 + 504);
  v86 = *(_QWORD *)(a1 + 512);
  v117 = v80;
  v121 = v81;
  v122 = v82;
  v123 = v83;
  v118 = v84;
  v119 = v85;
  v120 = v86;
  sub_108B970(&v117, &v121);
  v87 = *(_QWORD *)(a1 + 472);
  if ( v87 )
  {
    v88 = *(__int64 **)(a1 + 512);
    v89 = *(_QWORD *)(a1 + 544) + 8LL;
    if ( v89 > (unsigned __int64)v88 )
    {
      do
      {
        v90 = *v88++;
        j_j___libc_free_0(v90, 480);
      }
      while ( v89 > (unsigned __int64)v88 );
      v87 = *(_QWORD *)(a1 + 472);
    }
    j_j___libc_free_0(v87, 8LL * *(_QWORD *)(a1 + 480));
  }
  v91 = *(_QWORD *)(a1 + 408);
  v92 = *(_QWORD *)(a1 + 440);
  v93 = *(_QWORD *)(a1 + 448);
  v94 = *(_QWORD *)(a1 + 456);
  v95 = *(_QWORD *)(a1 + 416);
  v124 = *(_QWORD *)(a1 + 464);
  v96 = *(_QWORD *)(a1 + 424);
  v97 = *(_QWORD *)(a1 + 432);
  v117 = v91;
  v121 = v92;
  v122 = v93;
  v123 = v94;
  v118 = v95;
  v119 = v96;
  v120 = v97;
  sub_108B970(&v117, &v121);
  v98 = *(_QWORD *)(a1 + 392);
  if ( v98 )
  {
    v99 = *(__int64 **)(a1 + 432);
    v100 = *(_QWORD *)(a1 + 464) + 8LL;
    if ( v100 > (unsigned __int64)v99 )
    {
      do
      {
        v101 = *v99++;
        j_j___libc_free_0(v101, 480);
      }
      while ( v100 > (unsigned __int64)v99 );
      v98 = *(_QWORD *)(a1 + 392);
    }
    j_j___libc_free_0(v98, 8LL * *(_QWORD *)(a1 + 400));
  }
  v102 = *(_QWORD *)(a1 + 328);
  v103 = *(_QWORD *)(a1 + 360);
  v104 = *(_QWORD *)(a1 + 368);
  v105 = *(_QWORD *)(a1 + 376);
  v106 = *(_QWORD *)(a1 + 336);
  v124 = *(_QWORD *)(a1 + 384);
  v107 = *(_QWORD *)(a1 + 344);
  v108 = *(_QWORD *)(a1 + 352);
  v117 = v102;
  v121 = v103;
  v122 = v104;
  v123 = v105;
  v118 = v106;
  v119 = v107;
  v120 = v108;
  sub_108B970(&v117, &v121);
  v109 = *(_QWORD *)(a1 + 312);
  if ( v109 )
  {
    v110 = *(__int64 **)(a1 + 352);
    v111 = *(_QWORD *)(a1 + 384) + 8LL;
    if ( v111 > (unsigned __int64)v110 )
    {
      do
      {
        v112 = *v110++;
        j_j___libc_free_0(v112, 480);
      }
      while ( v111 > (unsigned __int64)v110 );
      v109 = *(_QWORD *)(a1 + 312);
    }
    j_j___libc_free_0(v109, 8LL * *(_QWORD *)(a1 + 320));
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 288), 16LL * *(unsigned int *)(a1 + 304), 8);
  v113 = 16LL * *(unsigned int *)(a1 + 272);
  sub_C7D6A0(*(_QWORD *)(a1 + 256), v113, 8);
  sub_C0BF30(a1 + 192);
  v114 = *(_QWORD *)(a1 + 184);
  if ( v114 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v114 + 8LL))(v114);
  v115 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)a1 = &unk_49E61E0;
  if ( v115 != a1 + 120 )
  {
    v113 = *(_QWORD *)(a1 + 120) + 1LL;
    j_j___libc_free_0(v115, v113);
  }
  return sub_E8EC10(a1, v113);
}
