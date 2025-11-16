// Function: sub_1F7A040
// Address: 0x1f7a040
//
_QWORD *__fastcall sub_1F7A040(
        __int64 a1,
        unsigned int a2,
        __int64 *a3,
        unsigned __int8 a4,
        int a5,
        double a6,
        double a7,
        __m128i a8)
{
  __int16 v8; // ax
  unsigned int v10; // r14d
  _BYTE *v12; // rdx
  __int128 *v13; // r15
  int v14; // eax
  __int64 v15; // rsi
  unsigned __int8 *v16; // r12
  const void **v17; // r8
  __int64 v18; // rcx
  __int64 *v19; // rax
  __int64 v20; // rax
  unsigned __int8 *v21; // r12
  const void **v22; // r8
  __int64 v23; // rcx
  __int64 v24; // r14
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // r15
  _QWORD *v27; // r10
  __int64 v28; // rsi
  _QWORD *v29; // r13
  unsigned int v31; // r15d
  __int64 v32; // r12
  char v33; // al
  _QWORD *v34; // r10
  __int64 v35; // r8
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned __int8 *v38; // r12
  const void **v39; // r15
  __int64 v40; // rcx
  __int64 v41; // r8
  unsigned __int64 v42; // r9
  __int64 v43; // r10
  __int64 v44; // rax
  __int64 v45; // rsi
  unsigned __int8 *v46; // r12
  const void **v47; // r8
  __int64 v48; // rcx
  __int64 v49; // r14
  __int64 v50; // rdx
  __int64 v51; // r15
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rdx
  unsigned int v55; // r15d
  __int64 v56; // r12
  __int64 v57; // r8
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  unsigned __int8 *v60; // r12
  const void **v61; // r15
  __int64 v62; // rcx
  __int64 v63; // r8
  unsigned __int64 v64; // r9
  _QWORD *v65; // r10
  __int64 v66; // rsi
  __int64 *v67; // rsi
  __int64 v68; // rsi
  unsigned __int8 *v69; // r12
  unsigned int v70; // ecx
  const void **v71; // r14
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rsi
  unsigned __int8 *v75; // r12
  const void **v76; // r8
  __int64 v77; // r15
  __int64 v78; // r10
  __int64 v79; // r11
  __int64 v80; // rcx
  __int64 v81; // rsi
  __int64 v82; // r8
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  unsigned __int8 *v85; // r12
  __int128 *v86; // r10
  __int64 v87; // r12
  __int64 v88; // rsi
  __int64 v89; // rbx
  __int64 v90; // rax
  __int64 v91; // r15
  __int64 v92; // rsi
  __int64 v93; // rdi
  __int64 v94; // rcx
  __int128 v95; // [rsp-10h] [rbp-A0h]
  __int128 v96; // [rsp-10h] [rbp-A0h]
  __int128 v97; // [rsp-10h] [rbp-A0h]
  __int64 v98; // [rsp+0h] [rbp-90h]
  __int64 v99; // [rsp+0h] [rbp-90h]
  __int64 v100; // [rsp+0h] [rbp-90h]
  __int64 v101; // [rsp+0h] [rbp-90h]
  __int64 v102; // [rsp+0h] [rbp-90h]
  __int64 v103; // [rsp+8h] [rbp-88h]
  __int64 v104; // [rsp+10h] [rbp-80h]
  _QWORD *v105; // [rsp+10h] [rbp-80h]
  __int64 v106; // [rsp+10h] [rbp-80h]
  __int64 v107; // [rsp+10h] [rbp-80h]
  __int64 v108; // [rsp+10h] [rbp-80h]
  unsigned int v109; // [rsp+10h] [rbp-80h]
  __int64 v110; // [rsp+10h] [rbp-80h]
  __int64 v111; // [rsp+10h] [rbp-80h]
  __int64 v112; // [rsp+10h] [rbp-80h]
  unsigned __int64 v113; // [rsp+18h] [rbp-78h]
  unsigned __int64 v114; // [rsp+18h] [rbp-78h]
  unsigned __int64 v115; // [rsp+18h] [rbp-78h]
  __int64 v116; // [rsp+20h] [rbp-70h]
  const void **v117; // [rsp+28h] [rbp-68h]
  _QWORD *v118; // [rsp+28h] [rbp-68h]
  const void **v119; // [rsp+28h] [rbp-68h]
  unsigned int v120; // [rsp+28h] [rbp-68h]
  __int64 v121; // [rsp+28h] [rbp-68h]
  const void **v122; // [rsp+28h] [rbp-68h]
  __int64 v123; // [rsp+28h] [rbp-68h]
  unsigned int v124; // [rsp+28h] [rbp-68h]
  _QWORD *v125; // [rsp+28h] [rbp-68h]
  void *v126; // [rsp+28h] [rbp-68h]
  const void **v127; // [rsp+28h] [rbp-68h]
  __int128 *v128; // [rsp+28h] [rbp-68h]
  __int64 v129; // [rsp+30h] [rbp-60h] BYREF
  int v130; // [rsp+38h] [rbp-58h]
  __int64 v131; // [rsp+40h] [rbp-50h] BYREF
  void *v132; // [rsp+48h] [rbp-48h] BYREF
  __int64 v133; // [rsp+50h] [rbp-40h]

  v8 = *(_WORD *)(a1 + 24);
  if ( v8 == 162 )
    return **(_QWORD ***)(a1 + 32);
  v10 = *(unsigned __int16 *)(a1 + 80);
  v12 = (_BYTE *)(*a3 + 792);
  if ( v8 > 79 )
  {
    if ( ((v8 - 157) & 0xFFF7) == 0 )
    {
      v44 = sub_1F7A040(**(_QWORD **)(a1 + 32), *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL), a3, a4, (unsigned int)(a5 + 1));
      v45 = *(_QWORD *)(a1 + 72);
      v46 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
      v47 = (const void **)*((_QWORD *)v46 + 1);
      v48 = *v46;
      v49 = v44;
      v51 = v50;
      v131 = v45;
      if ( v45 )
      {
        v107 = v48;
        v122 = v47;
        sub_1623A60((__int64)&v131, v45, 2);
        v48 = v107;
        v47 = v122;
      }
      v52 = *(unsigned __int16 *)(a1 + 24);
      *((_QWORD *)&v96 + 1) = v51;
      *(_QWORD *)&v96 = v49;
      LODWORD(v132) = *(_DWORD *)(a1 + 64);
      v53 = sub_1D309E0(a3, v52, (__int64)&v131, v48, v47, 0, a6, a7, *(double *)a8.m128i_i64, v96);
      v28 = v131;
      v29 = (_QWORD *)v53;
      if ( v131 )
        goto LABEL_16;
      return v29;
    }
    v118 = *(_QWORD **)(a1 + 32);
    v20 = sub_1F7A040(*v118, v118[1], a3, a4, (unsigned int)(a5 + 1));
    v21 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
    v22 = (const void **)*((_QWORD *)v21 + 1);
    v23 = *v21;
    v24 = v20;
    v26 = v25;
    v131 = *(_QWORD *)(a1 + 72);
    v27 = v118;
    if ( v131 )
    {
      v98 = v23;
      v105 = v118;
      v119 = v22;
      sub_1623A60((__int64)&v131, v131, 2);
      v23 = v98;
      v27 = v105;
      v22 = v119;
    }
    LODWORD(v132) = *(_DWORD *)(a1 + 64);
    v19 = sub_1D332F0(a3, 154, (__int64)&v131, v23, v22, 0, a6, a7, a8, v24, v26, *(_OWORD *)(v27 + 5));
  }
  else if ( v8 > 77 )
  {
    v55 = a4;
    v124 = a5 + 1;
    v56 = 16LL * a2;
    if ( (unsigned __int8)sub_1F79A30(
                            **(_QWORD **)(a1 + 32),
                            *(_DWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
                            a4,
                            a3[2],
                            v12,
                            a5 + 1,
                            a6,
                            a7,
                            *(double *)a8.m128i_i64) )
    {
      v57 = v124;
      v125 = *(_QWORD **)(a1 + 32);
      v58 = sub_1F7A040(*v125, v125[1], a3, v55, v57);
      v60 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + v56);
      v61 = (const void **)*((_QWORD *)v60 + 1);
      v62 = *v60;
      v63 = v58;
      v64 = v59;
      v131 = *(_QWORD *)(a1 + 72);
      v65 = v125;
      if ( v131 )
      {
        v114 = v59;
        v100 = v62;
        v108 = v58;
        sub_1623A60((__int64)&v131, v131, 2);
        v62 = v100;
        v63 = v108;
        v64 = v114;
        v65 = v125;
      }
      v66 = *(unsigned __int16 *)(a1 + 24);
      LODWORD(v132) = *(_DWORD *)(a1 + 64);
      v19 = sub_1D332F0(a3, v66, (__int64)&v131, v62, v61, v10, a6, a7, a8, v63, v64, *(_OWORD *)(v65 + 5));
    }
    else
    {
      v72 = sub_1F7A040(
              *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL),
              *(_QWORD *)(*(_QWORD *)(a1 + 32) + 48LL),
              a3,
              v55,
              v124);
      v74 = *(_QWORD *)(a1 + 72);
      v75 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + v56);
      v76 = (const void **)*((_QWORD *)v75 + 1);
      v77 = *v75;
      v78 = v72;
      v79 = v73;
      v131 = v74;
      v80 = *(_QWORD *)(a1 + 32);
      if ( v74 )
      {
        v103 = v73;
        v110 = *(_QWORD *)(a1 + 32);
        v127 = v76;
        v101 = v72;
        sub_1623A60((__int64)&v131, v74, 2);
        v78 = v101;
        v79 = v103;
        v80 = v110;
        v76 = v127;
      }
      v81 = *(unsigned __int16 *)(a1 + 24);
      *((_QWORD *)&v97 + 1) = v79;
      *(_QWORD *)&v97 = v78;
      LODWORD(v132) = *(_DWORD *)(a1 + 64);
      v19 = sub_1D332F0(
              a3,
              v81,
              (__int64)&v131,
              (unsigned int)v77,
              v76,
              v10,
              a6,
              a7,
              a8,
              *(_QWORD *)v80,
              *(_QWORD *)(v80 + 8),
              v97);
    }
  }
  else if ( v8 == 76 )
  {
    v31 = a4;
    v120 = a5 + 1;
    v32 = 16LL * a2;
    v33 = sub_1F79A30(
            **(_QWORD **)(a1 + 32),
            *(_DWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
            a4,
            a3[2],
            v12,
            a5 + 1,
            a6,
            a7,
            *(double *)a8.m128i_i64);
    v34 = *(_QWORD **)(a1 + 32);
    if ( v33 )
    {
      v35 = v120;
      v121 = *(_QWORD *)(a1 + 32);
      v36 = sub_1F7A040(*v34, v34[1], a3, v31, v35);
      v38 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + v32);
      v39 = (const void **)*((_QWORD *)v38 + 1);
      v40 = *v38;
      v41 = v36;
      v42 = v37;
      v131 = *(_QWORD *)(a1 + 72);
      v43 = v121;
      if ( v131 )
      {
        v113 = v37;
        v99 = v40;
        v106 = v36;
        sub_1623A60((__int64)&v131, v131, 2);
        v40 = v99;
        v41 = v106;
        v42 = v113;
        v43 = v121;
      }
      LODWORD(v132) = *(_DWORD *)(a1 + 64);
      v95 = *(_OWORD *)(v43 + 40);
    }
    else
    {
      v82 = v120;
      v128 = *(__int128 **)(a1 + 32);
      v83 = sub_1F7A040(v34[5], v34[6], a3, v31, v82);
      v85 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + v32);
      v39 = (const void **)*((_QWORD *)v85 + 1);
      v40 = *v85;
      v41 = v83;
      v42 = v84;
      v131 = *(_QWORD *)(a1 + 72);
      v86 = v128;
      if ( v131 )
      {
        v115 = v84;
        v102 = v40;
        v111 = v83;
        sub_1623A60((__int64)&v131, v131, 2);
        v40 = v102;
        v41 = v111;
        v42 = v115;
        v86 = v128;
      }
      LODWORD(v132) = *(_DWORD *)(a1 + 64);
      v95 = *v86;
    }
    v19 = sub_1D332F0(a3, 77, (__int64)&v131, v40, v39, v10, a6, a7, a8, v41, v42, v95);
  }
  else
  {
    if ( v8 != 77 )
    {
      v67 = (__int64 *)(*(_QWORD *)(a1 + 88) + 32LL);
      v126 = sub_16982C0();
      if ( (void *)*v67 == v126 )
        sub_169C6E0(&v132, (__int64)v67);
      else
        sub_16986C0(&v132, v67);
      if ( v132 == v126 )
        sub_169C8D0((__int64)&v132, a6, a7, *(double *)a8.m128i_i64);
      else
        sub_1699490((__int64)&v132);
      v68 = *(_QWORD *)(a1 + 72);
      v69 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
      v70 = *v69;
      v71 = (const void **)*((_QWORD *)v69 + 1);
      v129 = v68;
      if ( v68 )
      {
        v109 = v70;
        sub_1623A60((__int64)&v129, v68, 2);
        v70 = v109;
      }
      v130 = *(_DWORD *)(a1 + 64);
      v29 = sub_1D36490((__int64)a3, (__int64)&v131, (__int64)&v129, v70, v71, 0, a6, a7, a8);
      if ( v129 )
        sub_161E7C0((__int64)&v129, v129);
      if ( v132 == v126 )
      {
        v87 = v133;
        if ( v133 )
        {
          v88 = 32LL * *(_QWORD *)(v133 - 8);
          v89 = v133 + v88;
          if ( v133 != v133 + v88 )
          {
            do
            {
              v89 -= 32;
              if ( v126 == *(void **)(v89 + 8) )
              {
                v90 = *(_QWORD *)(v89 + 16);
                v116 = v90;
                if ( v90 )
                {
                  v91 = v90 + 32LL * *(_QWORD *)(v90 - 8);
                  if ( v90 != v91 )
                  {
                    do
                    {
                      v91 -= 32;
                      if ( v126 == *(void **)(v91 + 8) )
                      {
                        v92 = *(_QWORD *)(v91 + 16);
                        if ( v92 )
                        {
                          v93 = 32LL * *(_QWORD *)(v92 - 8);
                          v94 = v92 + v93;
                          if ( v92 != v92 + v93 )
                          {
                            do
                            {
                              v112 = v94 - 32;
                              sub_127D120((_QWORD *)(v94 - 24));
                              v94 = v112;
                            }
                            while ( v92 != v112 );
                          }
                          j_j_j___libc_free_0_0(v92 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v91 + 8);
                      }
                    }
                    while ( v116 != v91 );
                  }
                  j_j_j___libc_free_0_0(v116 - 8);
                }
              }
              else
              {
                sub_1698460(v89 + 8);
              }
            }
            while ( v87 != v89 );
          }
          j_j_j___libc_free_0_0(v87 - 8);
        }
      }
      else
      {
        sub_1698460((__int64)&v132);
      }
      return v29;
    }
    v13 = *(__int128 **)(a1 + 32);
    v14 = *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL);
    if ( v14 == 11 || v14 == 33 )
    {
      v123 = *(_QWORD *)(*(_QWORD *)v13 + 88LL);
      v54 = *(void **)(v123 + 32) == sub_16982C0() ? *(_QWORD *)(v123 + 40) + 8LL : v123 + 32;
      if ( (*(_BYTE *)(v54 + 18) & 7) == 3 )
        return (_QWORD *)*((_QWORD *)v13 + 5);
    }
    v15 = *(_QWORD *)(a1 + 72);
    v16 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
    v17 = (const void **)*((_QWORD *)v16 + 1);
    v18 = *v16;
    v131 = v15;
    if ( v15 )
    {
      v104 = v18;
      v117 = v17;
      sub_1623A60((__int64)&v131, v15, 2);
      v18 = v104;
      v17 = v117;
    }
    LODWORD(v132) = *(_DWORD *)(a1 + 64);
    v19 = sub_1D332F0(
            a3,
            77,
            (__int64)&v131,
            v18,
            v17,
            v10,
            a6,
            a7,
            a8,
            *((_QWORD *)v13 + 5),
            *((_QWORD *)v13 + 6),
            *v13);
  }
  v28 = v131;
  v29 = v19;
  if ( v131 )
LABEL_16:
    sub_161E7C0((__int64)&v131, v28);
  return v29;
}
