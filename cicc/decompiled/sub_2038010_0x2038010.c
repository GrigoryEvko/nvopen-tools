// Function: sub_2038010
// Address: 0x2038010
//
__int64 __fastcall sub_2038010(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r14d
  const __m128i *v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rax
  char v13; // dl
  unsigned int v14; // eax
  const void **v15; // rdx
  _QWORD *v16; // r15
  unsigned __int8 v17; // bl
  unsigned int v18; // eax
  unsigned __int8 v19; // r15
  __int64 v20; // rcx
  int v21; // r8d
  int v22; // r9d
  __int64 v23; // r15
  unsigned int v24; // ebx
  __int64 *v25; // rax
  __int64 v26; // rbx
  __int64 *v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // r9
  unsigned int v30; // eax
  const void **v31; // rdx
  __int64 v32; // rbx
  __int64 *v33; // rax
  int v34; // edx
  int v35; // edi
  __int64 *v36; // rdx
  __int64 *v37; // rax
  __int64 *v38; // r15
  __int64 v39; // rax
  unsigned int v40; // edx
  char v41; // al
  __int128 v42; // rax
  __int128 v43; // rax
  __int64 *v44; // rdi
  __int64 v45; // rax
  int v46; // edx
  int v47; // edi
  __int64 v48; // rdx
  __int64 *v49; // rax
  _QWORD *v50; // rdi
  _QWORD *v51; // rbx
  int v52; // edx
  int v53; // r14d
  __int64 v54; // rdx
  __int64 *v55; // rcx
  __int64 *v56; // rax
  unsigned __int64 v57; // rdi
  __int64 v58; // r14
  const void **v60; // rdx
  __int64 *v61; // rax
  __int64 v62; // rsi
  unsigned __int32 v63; // edx
  unsigned __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r8
  __int64 v69; // r9
  int v70; // r15d
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 *v75; // r14
  __int64 (__fastcall *v76)(__int64, __int64); // rbx
  __int64 v77; // rax
  unsigned int v78; // edx
  unsigned __int8 v79; // al
  __int128 v80; // rax
  __int128 v81; // rax
  __int64 *v82; // rdi
  __int64 *v83; // rax
  __int64 *v84; // rdi
  __int64 *v85; // rax
  __int64 v86; // r8
  __int64 v87; // r9
  _QWORD *v88; // rax
  int v89; // edx
  int v90; // edi
  _QWORD *v91; // rsi
  unsigned int v92; // edx
  __int64 v93; // rax
  __int64 *v94; // rax
  __int128 v95; // rax
  __int64 v96; // rax
  __int128 v97; // [rsp-10h] [rbp-250h]
  __int128 v98; // [rsp-10h] [rbp-250h]
  int v99; // [rsp+0h] [rbp-240h]
  int v100; // [rsp+0h] [rbp-240h]
  unsigned int v101; // [rsp+8h] [rbp-238h]
  __int64 v102; // [rsp+8h] [rbp-238h]
  __int64 v103; // [rsp+8h] [rbp-238h]
  __int64 v104; // [rsp+8h] [rbp-238h]
  int v105; // [rsp+10h] [rbp-230h]
  int v106; // [rsp+10h] [rbp-230h]
  unsigned __int16 v107; // [rsp+16h] [rbp-22Ah]
  const void **v108; // [rsp+18h] [rbp-228h]
  __int64 v109; // [rsp+18h] [rbp-228h]
  __int64 v110; // [rsp+18h] [rbp-228h]
  unsigned int v111; // [rsp+20h] [rbp-220h]
  unsigned int v112; // [rsp+28h] [rbp-218h]
  unsigned int v113; // [rsp+2Ch] [rbp-214h]
  unsigned int v114; // [rsp+30h] [rbp-210h]
  const void **v115; // [rsp+38h] [rbp-208h]
  unsigned int v116; // [rsp+40h] [rbp-200h]
  const void **v117; // [rsp+48h] [rbp-1F8h]
  __int64 v118; // [rsp+50h] [rbp-1F0h]
  unsigned int v119; // [rsp+58h] [rbp-1E8h]
  unsigned int v120; // [rsp+58h] [rbp-1E8h]
  __int64 v121; // [rsp+58h] [rbp-1E8h]
  unsigned __int32 v122; // [rsp+60h] [rbp-1E0h]
  __int64 (__fastcall *v123)(__int64, __int64); // [rsp+60h] [rbp-1E0h]
  __int64 v124; // [rsp+60h] [rbp-1E0h]
  __int128 v126; // [rsp+70h] [rbp-1D0h]
  __int64 v127; // [rsp+C0h] [rbp-180h] BYREF
  int v128; // [rsp+C8h] [rbp-178h]
  __int64 v129; // [rsp+D0h] [rbp-170h] BYREF
  const void **v130; // [rsp+D8h] [rbp-168h]
  unsigned int v131; // [rsp+E0h] [rbp-160h] BYREF
  __int64 v132; // [rsp+E8h] [rbp-158h]
  __m128i v133; // [rsp+F0h] [rbp-150h] BYREF
  __int64 *v134; // [rsp+100h] [rbp-140h] BYREF
  __int64 v135; // [rsp+108h] [rbp-138h]
  _QWORD v136[38]; // [rsp+110h] [rbp-130h] BYREF

  v7 = *(const __m128i **)(a2 + 32);
  v8 = _mm_loadu_si128(v7);
  v118 = v7->m128i_i64[0];
  v9 = *(_QWORD *)(a2 + 72);
  v122 = v7->m128i_u32[2];
  v127 = v9;
  *((_QWORD *)&v126 + 1) = v8.m128i_i64[1];
  if ( v9 )
    sub_1623A60((__int64)&v127, v9, 2);
  v10 = *(_QWORD *)a1;
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
  v128 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v134, v10, v11, **(unsigned __int8 **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v129) = v135;
  v130 = (const void **)v136[0];
  if ( (_BYTE)v135 )
    v112 = word_4305480[(unsigned __int8)(v135 - 14)];
  else
    v112 = sub_1F58D30((__int64)&v129);
  v12 = *(_QWORD *)(v118 + 40) + 16LL * v122;
  v13 = *(_BYTE *)v12;
  v132 = *(_QWORD *)(v12 + 8);
  LOBYTE(v131) = v13;
  LOBYTE(v14) = sub_1F7E0F0((__int64)&v131);
  v115 = v15;
  v114 = v14;
  v16 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  v108 = 0;
  v17 = sub_1D15020(v14, v112);
  if ( !v17 )
  {
    v119 = sub_1F593D0(v16, v114, (__int64)v115, v112);
    v17 = v119;
    v108 = v60;
  }
  v18 = v119;
  v19 = v131;
  LOBYTE(v18) = v17;
  v120 = v18;
  v113 = *(unsigned __int16 *)(a2 + 24);
  if ( (_BYTE)v131 )
    v111 = word_4305480[(unsigned __int8)(v131 - 14)];
  else
    v111 = sub_1F58D30((__int64)&v131);
  v107 = *(_WORD *)(a2 + 80);
  sub_1F40D10((__int64)&v134, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v19, v132);
  if ( (_BYTE)v134 == 7 )
  {
    v61 = *(__int64 **)(a2 + 32);
    v62 = *v61;
    v118 = sub_20363F0(a1, *v61, v61[1]);
    *(_QWORD *)&v126 = v118;
    v122 = v63;
    v64 = v63 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v65 = *(_QWORD *)(v118 + 40) + 16LL * v63;
    v102 = v64;
    *((_QWORD *)&v126 + 1) = v64;
    v66 = *(_QWORD *)(v65 + 8);
    LOBYTE(v131) = *(_BYTE *)v65;
    v132 = v66;
    v111 = sub_1D15970(&v131);
    if ( v111 == v112 )
    {
      v84 = *(__int64 **)(a1 + 8);
      if ( *(_DWORD *)(a2 + 56) != 1 )
      {
        v83 = sub_1D332F0(
                v84,
                v113,
                (__int64)&v127,
                (unsigned int)v129,
                v130,
                v107,
                *(double *)v8.m128i_i64,
                a4,
                a5,
                v118,
                *((unsigned __int64 *)&v126 + 1),
                *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
        goto LABEL_62;
      }
      v96 = sub_1D309E0(
              v84,
              v113,
              (__int64)&v127,
              (unsigned int)v129,
              v130,
              0,
              *(double *)v8.m128i_i64,
              a4,
              *(double *)a5.m128i_i64,
              v126);
LABEL_76:
      v58 = v96;
      goto LABEL_39;
    }
    v70 = sub_1D159A0((char *)&v129, v62, v67, v112, v68, v69, v99, v102, v105, (__int64)v108);
    if ( v70 == (unsigned int)sub_1D159A0((char *)&v131, v62, v71, v72, v73, v74, v100, v103, v106, v110) )
    {
      if ( v113 == 142 )
      {
        v58 = sub_1D327E0(
                *(__int64 **)(a1 + 8),
                v118,
                v104,
                (__int64)&v127,
                v129,
                v130,
                *(double *)v8.m128i_i64,
                a4,
                *(double *)a5.m128i_i64);
        goto LABEL_39;
      }
      if ( v113 == 143 )
      {
        v58 = sub_1D32810(
                *(__int64 **)(a1 + 8),
                v118,
                v104,
                (__int64)&v127,
                v129,
                v130,
                *(double *)v8.m128i_i64,
                a4,
                *(double *)a5.m128i_i64);
        goto LABEL_39;
      }
    }
  }
  if ( !v17 || (v23 = *(_QWORD *)a1, v22 = v17, !*(_QWORD *)(*(_QWORD *)a1 + 8LL * v17 + 120)) )
  {
LABEL_14:
    v25 = v136;
    v134 = v136;
    v135 = 0x1000000000LL;
    if ( v112 > 0x10 )
    {
      sub_16CD150((__int64)&v134, v136, v112, 16, v21, v22);
      v25 = v134;
    }
    v26 = 2LL * v112;
    v27 = &v25[v26];
    for ( LODWORD(v135) = v112; v27 != v25; v25 += 2 )
    {
      if ( v25 )
      {
        *v25 = 0;
        *((_DWORD *)v25 + 2) = 0;
      }
    }
    LOBYTE(v28) = sub_1F7E0F0((__int64)&v129);
    v116 = v28;
    v30 = v111;
    v117 = v31;
    if ( v111 > v112 )
      v30 = v112;
    v101 = v30;
    if ( v30 )
    {
      v32 = 0;
      v109 = v122;
      do
      {
        while ( 1 )
        {
          v38 = *(__int64 **)(a1 + 8);
          v121 = *(_QWORD *)a1;
          v123 = *(__int64 (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 48LL);
          v39 = sub_1E0A0C0(v38[4]);
          if ( v123 == sub_1D13A20 )
          {
            v40 = 8 * sub_15A9520(v39, 0);
            if ( v40 == 32 )
            {
              v41 = 5;
            }
            else if ( v40 > 0x20 )
            {
              v41 = 6;
              if ( v40 != 64 )
              {
                v41 = 0;
                if ( v40 == 128 )
                  v41 = 7;
              }
            }
            else
            {
              v41 = 3;
              if ( v40 != 8 )
                v41 = 4 * (v40 == 16);
            }
          }
          else
          {
            v41 = v123(v121, v39);
          }
          LOBYTE(v5) = v41;
          *(_QWORD *)&v42 = sub_1D38BB0((__int64)v38, v32, (__int64)&v127, v5, 0, 0, v8, a4, a5, 0);
          *((_QWORD *)&v126 + 1) = v109 | *((_QWORD *)&v126 + 1) & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v43 = sub_1D332F0(
                              v38,
                              106,
                              (__int64)&v127,
                              v114,
                              v115,
                              0,
                              *(double *)v8.m128i_i64,
                              a4,
                              a5,
                              v118,
                              *((unsigned __int64 *)&v126 + 1),
                              v42);
          v44 = *(__int64 **)(a1 + 8);
          v124 = 2 * v32;
          if ( *(_DWORD *)(a2 + 56) == 1 )
            break;
          ++v32;
          v33 = sub_1D332F0(
                  v44,
                  v113,
                  (__int64)&v127,
                  v116,
                  v117,
                  v107,
                  *(double *)v8.m128i_i64,
                  a4,
                  a5,
                  v43,
                  *((unsigned __int64 *)&v43 + 1),
                  *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
          v35 = v34;
          v36 = v33;
          v37 = v134;
          v134[v124] = (__int64)v36;
          LODWORD(v37[v124 + 1]) = v35;
          if ( v101 == v32 )
            goto LABEL_32;
        }
        ++v32;
        v45 = sub_1D309E0(
                v44,
                v113,
                (__int64)&v127,
                v116,
                v117,
                0,
                *(double *)v8.m128i_i64,
                a4,
                *(double *)a5.m128i_i64,
                v43);
        v47 = v46;
        v48 = v45;
        v49 = v134;
        v134[v124] = v48;
        LODWORD(v49[v124 + 1]) = v47;
      }
      while ( v101 != v32 );
    }
LABEL_32:
    v50 = *(_QWORD **)(a1 + 8);
    v133.m128i_i64[0] = 0;
    v133.m128i_i32[2] = 0;
    v51 = sub_1D2B300(v50, 0x30u, (__int64)&v133, v116, (__int64)v117, v29);
    v53 = v52;
    if ( v133.m128i_i64[0] )
      sub_161E7C0((__int64)&v133, v133.m128i_i64[0]);
    if ( v101 < v112 )
    {
      v54 = 2LL * v101;
      do
      {
        v55 = v134;
        v134[v54] = (__int64)v51;
        LODWORD(v55[v54 + 1]) = v53;
        v54 += 2;
      }
      while ( v54 != 2 * (v101 + (unsigned __int64)(v112 - 1 - v101) + 1) );
    }
    *((_QWORD *)&v97 + 1) = (unsigned int)v135;
    *(_QWORD *)&v97 = v134;
    v56 = sub_1D359D0(*(__int64 **)(a1 + 8), 104, (__int64)&v127, v129, v130, 0, *(double *)v8.m128i_i64, a4, a5, v97);
    v57 = (unsigned __int64)v134;
    v58 = (__int64)v56;
    if ( v134 != v136 )
      goto LABEL_38;
    goto LABEL_39;
  }
  v24 = v112 / v111;
  if ( v112 % v111 )
  {
    if ( v111 % v112 )
      goto LABEL_14;
    v75 = *(__int64 **)(a1 + 8);
    v76 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 48LL);
    v77 = sub_1E0A0C0(v75[4]);
    if ( v76 == sub_1D13A20 )
    {
      v78 = 8 * sub_15A9520(v77, 0);
      if ( v78 == 32 )
      {
        v79 = 5;
      }
      else if ( v78 > 0x20 )
      {
        v79 = 6;
        if ( v78 != 64 )
        {
          v79 = 0;
          if ( v78 == 128 )
            v79 = 7;
        }
      }
      else
      {
        v79 = 3;
        if ( v78 != 8 )
          v79 = 4 * (v78 == 16);
      }
    }
    else
    {
      v79 = v76(v23, v77);
    }
    *(_QWORD *)&v80 = sub_1D38BB0((__int64)v75, 0, (__int64)&v127, v79, 0, 0, v8, a4, a5, 0);
    *(_QWORD *)&v81 = sub_1D332F0(
                        v75,
                        109,
                        (__int64)&v127,
                        v120,
                        v108,
                        0,
                        *(double *)v8.m128i_i64,
                        a4,
                        a5,
                        v118,
                        v122 | *((_QWORD *)&v126 + 1) & 0xFFFFFFFF00000000LL,
                        v80);
    v82 = *(__int64 **)(a1 + 8);
    if ( *(_DWORD *)(a2 + 56) != 1 )
    {
      v83 = sub_1D332F0(
              v82,
              v113,
              (__int64)&v127,
              (unsigned int)v129,
              v130,
              v107,
              *(double *)v8.m128i_i64,
              a4,
              a5,
              v81,
              *((unsigned __int64 *)&v81 + 1),
              *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
LABEL_62:
      v58 = (__int64)v83;
      goto LABEL_39;
    }
    v96 = sub_1D309E0(
            v82,
            v113,
            (__int64)&v127,
            (unsigned int)v129,
            v130,
            0,
            *(double *)v8.m128i_i64,
            a4,
            *(double *)a5.m128i_i64,
            v81);
    goto LABEL_76;
  }
  v134 = v136;
  v135 = 0x1000000000LL;
  v133.m128i_i64[0] = 0;
  v133.m128i_i32[2] = 0;
  sub_202F910((__int64)&v134, v24, &v133, v20, v21, v22);
  v85 = v134;
  *v134 = v118;
  *((_DWORD *)v85 + 2) = v122;
  v88 = sub_1D2B530(*(_QWORD **)(a1 + 8), v131, v132, v122, v86, v87);
  v90 = v89;
  v91 = v88;
  v92 = 1;
  if ( v24 != 1 )
  {
    do
    {
      v93 = v92++;
      v94 = &v134[2 * v93];
      *v94 = (__int64)v91;
      *((_DWORD *)v94 + 2) = v90;
    }
    while ( v24 != v92 );
  }
  *((_QWORD *)&v98 + 1) = (unsigned int)v135;
  *(_QWORD *)&v98 = v134;
  *(_QWORD *)&v95 = sub_1D359D0(
                      *(__int64 **)(a1 + 8),
                      107,
                      (__int64)&v127,
                      v120,
                      v108,
                      0,
                      *(double *)v8.m128i_i64,
                      a4,
                      a5,
                      v98);
  if ( *(_DWORD *)(a2 + 56) == 1 )
    v58 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            v113,
            (__int64)&v127,
            (unsigned int)v129,
            v130,
            0,
            *(double *)v8.m128i_i64,
            a4,
            *(double *)a5.m128i_i64,
            v95);
  else
    v58 = (__int64)sub_1D332F0(
                     *(__int64 **)(a1 + 8),
                     v113,
                     (__int64)&v127,
                     (unsigned int)v129,
                     v130,
                     v107,
                     *(double *)v8.m128i_i64,
                     a4,
                     a5,
                     v95,
                     *((unsigned __int64 *)&v95 + 1),
                     *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  v57 = (unsigned __int64)v134;
  if ( v134 != v136 )
LABEL_38:
    _libc_free(v57);
LABEL_39:
  if ( v127 )
    sub_161E7C0((__int64)&v127, v127);
  return v58;
}
