// Function: sub_20B42D0
// Address: 0x20b42d0
//
__int64 *__fastcall sub_20B42D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        char a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  unsigned __int8 *v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rax
  const void **v15; // rdx
  unsigned int v16; // ebx
  const __m128i *v17; // roff
  unsigned int v19; // r10d
  unsigned int v21; // r14d
  unsigned int v22; // edx
  _QWORD *v23; // r13
  unsigned int v24; // eax
  __int64 *v25; // r12
  __int64 v27; // rax
  __int128 v28; // rax
  __int128 *v29; // rbx
  const void ***v30; // rax
  int v31; // edx
  __int64 v32; // r9
  __int64 *v33; // rax
  int v34; // r8d
  int v35; // r9d
  __int64 *v36; // rbx
  __int64 v37; // rax
  unsigned int v38; // esi
  unsigned int v39; // edx
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // r13
  unsigned __int8 *v43; // rax
  __int64 v44; // rsi
  unsigned int v45; // eax
  __int64 v46; // rdx
  const void **v47; // r13
  unsigned int v48; // r12d
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  int v52; // eax
  __int128 v53; // rax
  __int64 *v54; // rax
  int v55; // r8d
  int v56; // r9d
  __int64 v57; // rdx
  __int64 v58; // r13
  __int64 *v59; // r12
  __int64 v60; // rdx
  __int64 *v61; // rbx
  __int64 v62; // rcx
  const void **v63; // r8
  __int64 v64; // rdi
  __int64 v65; // rbx
  __int128 v66; // rax
  int v67; // r8d
  int v68; // r9d
  __int64 v69; // rax
  const __m128i *v70; // roff
  __int128 v71; // rax
  unsigned int v72; // edx
  __int64 v73; // r13
  __int64 v74; // rax
  __int64 v75; // rax
  const void **v76; // rdx
  __int128 v77; // rax
  int v78; // r8d
  int v79; // r9d
  __int64 v80; // rax
  unsigned __int32 v81; // edx
  __int128 v82; // rax
  __int64 v83; // rax
  int v84; // eax
  __int64 v85; // rax
  int v86; // r8d
  int v87; // r9d
  __int64 *v88; // r14
  unsigned int v89; // edx
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 *v92; // rax
  int v93; // r8d
  int v94; // r9d
  unsigned int v95; // edx
  unsigned __int64 v96; // rcx
  __int64 v97; // rdx
  unsigned __int8 *v98; // rax
  __int64 v99; // rax
  const void **v100; // rdx
  __int128 v101; // rax
  int v102; // r8d
  int v103; // r9d
  __int64 *v104; // r14
  unsigned int v105; // edx
  __int64 v106; // rax
  __int128 v107; // [rsp-10h] [rbp-110h]
  unsigned __int32 v108; // [rsp+Ch] [rbp-F4h]
  __int64 v109; // [rsp+10h] [rbp-F0h]
  __int64 *v110; // [rsp+10h] [rbp-F0h]
  __m128 v111; // [rsp+20h] [rbp-E0h]
  __int128 v112; // [rsp+20h] [rbp-E0h]
  unsigned int v113; // [rsp+20h] [rbp-E0h]
  __int64 v114; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v115; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v116; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v117; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v118; // [rsp+38h] [rbp-C8h]
  unsigned int v121; // [rsp+48h] [rbp-B8h]
  unsigned int v122; // [rsp+50h] [rbp-B0h] BYREF
  const void **v123; // [rsp+58h] [rbp-A8h]
  __int64 v124; // [rsp+60h] [rbp-A0h] BYREF
  int v125; // [rsp+68h] [rbp-98h]
  unsigned __int64 v126; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v127; // [rsp+78h] [rbp-88h]
  unsigned __int64 v128; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v129; // [rsp+88h] [rbp-78h]
  unsigned __int64 v130; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v131; // [rsp+98h] [rbp-68h]
  unsigned __int64 v132; // [rsp+A0h] [rbp-60h] BYREF
  unsigned int v133; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v134; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v135; // [rsp+B8h] [rbp-48h]
  unsigned int v136; // [rsp+C0h] [rbp-40h]

  v12 = *(unsigned __int8 **)(a2 + 40);
  v13 = *(_QWORD *)(a2 + 72);
  v14 = *v12;
  v15 = (const void **)*((_QWORD *)v12 + 1);
  v124 = v13;
  LOBYTE(v122) = v14;
  v123 = v15;
  if ( v13 )
  {
    sub_1623A60((__int64)&v124, v13, 2);
    v14 = (unsigned __int8)v122;
  }
  v125 = *(_DWORD *)(a2 + 64);
  if ( !(_BYTE)v14 || !*(_QWORD *)(a1 + 8 * v14 + 120) )
  {
    v25 = 0;
    goto LABEL_32;
  }
  if ( (*(_BYTE *)(a2 + 80) & 8) == 0 )
  {
    sub_16AB900(&v134, a3);
    if ( a5 )
    {
      if ( (_BYTE)v122 == 1 )
      {
        v27 = 1;
        if ( *(_BYTE *)(a1 + 2794) )
        {
LABEL_38:
          if ( !*(_BYTE *)(a1 + 259 * v27 + 2481) )
          {
LABEL_39:
            *(_QWORD *)&v28 = sub_1D38970(
                                (__int64)a4,
                                (__int64)&v134,
                                (__int64)&v124,
                                v122,
                                v123,
                                0,
                                a7,
                                *(double *)a8.m128i_i64,
                                a9,
                                0);
            v29 = *(__int128 **)(a2 + 32);
            v112 = v28;
            v30 = (const void ***)sub_1D252B0((__int64)a4, v122, (__int64)v123, v122, (__int64)v123);
            v33 = sub_1D37440(
                    a4,
                    59,
                    (__int64)&v124,
                    v30,
                    v31,
                    v32,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    *v29,
                    v112);
            v113 = 1;
            v36 = v33;
            goto LABEL_40;
          }
LABEL_81:
          v25 = 0;
          goto LABEL_50;
        }
      }
      else
      {
        if ( !(_BYTE)v122 || !*(_QWORD *)(a1 + 8LL * (unsigned __int8)v122 + 120) )
          goto LABEL_81;
        if ( *(_BYTE *)(a1 + 259LL * (unsigned __int8)v122 + 2535) )
        {
          if ( !*(_QWORD *)(a1 + 8 * ((unsigned __int8)v122 + 14LL) + 8) )
            goto LABEL_81;
          v27 = (unsigned __int8)v122;
          goto LABEL_38;
        }
      }
LABEL_71:
      *(_QWORD *)&v71 = sub_1D38970(
                          (__int64)a4,
                          (__int64)&v134,
                          (__int64)&v124,
                          v122,
                          v123,
                          0,
                          a7,
                          *(double *)a8.m128i_i64,
                          a9,
                          0);
      v36 = sub_1D332F0(
              a4,
              113,
              (__int64)&v124,
              v122,
              v123,
              0,
              *(double *)a7.m128i_i64,
              *(double *)a8.m128i_i64,
              a9,
              **(_QWORD **)(a2 + 32),
              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
              v71);
      v113 = v72;
      v115 = v72;
LABEL_40:
      v37 = *(unsigned int *)(a6 + 8);
      if ( (unsigned int)v37 >= *(_DWORD *)(a6 + 12) )
      {
        sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v34, v35);
        v37 = *(unsigned int *)(a6 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * v37) = v36;
      ++*(_DWORD *)(a6 + 8);
      v38 = *(_DWORD *)(a3 + 8);
      v39 = v38 - 1;
      v40 = 1LL << ((unsigned __int8)v38 - 1);
      if ( v38 > 0x40 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)a3 + 8LL * (v39 >> 6)) & v40) != 0 )
          goto LABEL_90;
        v84 = sub_16A57B0(a3);
        v40 = 1LL << ((unsigned __int8)v38 - 1);
        v39 = v38 - 1;
        if ( v38 == v84 )
        {
LABEL_88:
          if ( v38 > 0x40 )
          {
            v41 = *(_QWORD *)(*(_QWORD *)a3 + 8LL * (v39 >> 6)) & v40;
LABEL_45:
            if ( !v41 )
            {
LABEL_46:
              v42 = sub_1E0A0C0(a4[4]);
              if ( v136 )
              {
                v98 = (unsigned __int8 *)(v36[5] + 16LL * v113);
                v99 = sub_1F40B60(a1, *v98, *((_QWORD *)v98 + 1), v42, 1);
                *(_QWORD *)&v101 = sub_1D38BB0(
                                     (__int64)a4,
                                     v136,
                                     (__int64)&v124,
                                     v99,
                                     v100,
                                     0,
                                     a7,
                                     *(double *)a8.m128i_i64,
                                     a9,
                                     0);
                v115 = v113 | v115 & 0xFFFFFFFF00000000LL;
                v104 = sub_1D332F0(
                         a4,
                         123,
                         (__int64)&v124,
                         v122,
                         v123,
                         0,
                         *(double *)a7.m128i_i64,
                         *(double *)a8.m128i_i64,
                         a9,
                         (__int64)v36,
                         v115,
                         v101);
                v36 = v104;
                v113 = v105;
                v106 = *(unsigned int *)(a6 + 8);
                if ( (unsigned int)v106 >= *(_DWORD *)(a6 + 12) )
                {
                  sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v102, v103);
                  v106 = *(unsigned int *)(a6 + 8);
                }
                *(_QWORD *)(*(_QWORD *)a6 + 8 * v106) = v104;
                ++*(_DWORD *)(a6 + 8);
              }
              v43 = (unsigned __int8 *)(v36[5] + 16LL * v113);
              v44 = *v43;
              v45 = sub_1F40B60(a1, v44, *((_QWORD *)v43 + 1), v42, 1);
              v47 = (const void **)v46;
              v48 = v45;
              v52 = sub_1D159C0((__int64)&v122, v44, v46, v49, v50, v51);
              *(_QWORD *)&v53 = sub_1D38BB0(
                                  (__int64)a4,
                                  (unsigned int)(v52 - 1),
                                  (__int64)&v124,
                                  v48,
                                  v47,
                                  0,
                                  a7,
                                  *(double *)a8.m128i_i64,
                                  a9,
                                  0);
              v114 = (__int64)v36;
              v116 = v113 | v115 & 0xFFFFFFFF00000000LL;
              v54 = sub_1D332F0(
                      a4,
                      124,
                      (__int64)&v124,
                      v122,
                      v123,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v36,
                      v116,
                      v53);
              v58 = v57;
              v59 = v54;
              v60 = *(unsigned int *)(a6 + 8);
              v61 = v54;
              if ( (unsigned int)v60 >= *(_DWORD *)(a6 + 12) )
              {
                sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v55, v56);
                v60 = *(unsigned int *)(a6 + 8);
              }
              *(_QWORD *)(*(_QWORD *)a6 + 8 * v60) = v61;
              v62 = v122;
              v63 = v123;
              *((_QWORD *)&v107 + 1) = v58;
              ++*(_DWORD *)(a6 + 8);
              *(_QWORD *)&v107 = v59;
              v25 = sub_1D332F0(
                      a4,
                      52,
                      (__int64)&v124,
                      v62,
                      v63,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v114,
                      v116,
                      v107);
LABEL_50:
              if ( v135 <= 0x40 )
                goto LABEL_32;
              v64 = v134;
              if ( !v134 )
                goto LABEL_32;
              goto LABEL_52;
            }
LABEL_90:
            v85 = 1LL << ((unsigned __int8)v135 - 1);
            if ( v135 > 0x40 )
            {
              if ( (*(_QWORD *)(v134 + 8LL * ((v135 - 1) >> 6)) & v85) != 0 )
                goto LABEL_46;
              v121 = v135;
              if ( v121 == (unsigned int)sub_16A57B0((__int64)&v134) )
                goto LABEL_46;
            }
            else if ( (v85 & v134) != 0 || !v134 )
            {
              goto LABEL_46;
            }
            v117 = v113 | v115 & 0xFFFFFFFF00000000LL;
            v88 = sub_1D332F0(
                    a4,
                    53,
                    (__int64)&v124,
                    v122,
                    v123,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    (__int64)v36,
                    v117,
                    *(_OWORD *)*(_QWORD *)(a2 + 32));
            v36 = v88;
            v113 = v89;
            v90 = *(unsigned int *)(a6 + 8);
            v115 = v89 | v117 & 0xFFFFFFFF00000000LL;
            if ( (unsigned int)v90 >= *(_DWORD *)(a6 + 12) )
            {
              sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v86, v87);
              v90 = *(unsigned int *)(a6 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a6 + 8 * v90) = v88;
            ++*(_DWORD *)(a6 + 8);
            goto LABEL_46;
          }
LABEL_44:
          v41 = v40 & *(_QWORD *)a3;
          goto LABEL_45;
        }
      }
      else if ( (*(_QWORD *)a3 & v40) != 0 || !*(_QWORD *)a3 )
      {
        goto LABEL_44;
      }
      v91 = v134;
      if ( v135 > 0x40 )
        v91 = *(_QWORD *)(v134 + 8LL * ((v135 - 1) >> 6));
      if ( (v91 & (1LL << ((unsigned __int8)v135 - 1))) != 0 )
      {
        v118 = v113 | v115 & 0xFFFFFFFF00000000LL;
        v92 = sub_1D332F0(
                a4,
                52,
                (__int64)&v124,
                v122,
                v123,
                0,
                *(double *)a7.m128i_i64,
                *(double *)a8.m128i_i64,
                a9,
                (__int64)v36,
                v118,
                *(_OWORD *)*(_QWORD *)(a2 + 32));
        v113 = v95;
        v36 = v92;
        v96 = v95 | v118 & 0xFFFFFFFF00000000LL;
        v97 = *(unsigned int *)(a6 + 8);
        v115 = v96;
        if ( (unsigned int)v97 >= *(_DWORD *)(a6 + 12) )
        {
          v110 = v92;
          sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v93, v94);
          v97 = *(unsigned int *)(a6 + 8);
          v92 = v110;
        }
        *(_QWORD *)(*(_QWORD *)a6 + 8 * v97) = v92;
        ++*(_DWORD *)(a6 + 8);
        v38 = *(_DWORD *)(a3 + 8);
        v39 = v38 - 1;
        v40 = 1LL << ((unsigned __int8)v38 - 1);
      }
      goto LABEL_88;
    }
    if ( (_BYTE)v122 == 1 )
    {
      v83 = 1;
      if ( (*(_BYTE *)(a1 + 2794) & 0xFB) == 0 )
        goto LABEL_71;
    }
    else
    {
      if ( !(_BYTE)v122 || !*(_QWORD *)(a1 + 8LL * (unsigned __int8)v122 + 120) )
        goto LABEL_81;
      if ( (*(_BYTE *)(a1 + 259LL * (unsigned __int8)v122 + 2535) & 0xFB) == 0 )
        goto LABEL_71;
      if ( !*(_QWORD *)(a1 + 8 * ((unsigned __int8)v122 + 14LL) + 8) )
        goto LABEL_81;
      v83 = (unsigned __int8)v122;
    }
    if ( (*(_BYTE *)(a1 + 259 * v83 + 2481) & 0xFB) == 0 )
      goto LABEL_39;
    goto LABEL_81;
  }
  v16 = *(_DWORD *)(a3 + 8);
  v127 = v16;
  if ( v16 <= 0x40 )
  {
    v126 = *(_QWORD *)a3;
    v17 = *(const __m128i **)(a2 + 32);
    a7 = _mm_loadu_si128(v17);
    v109 = v17->m128i_i64[0];
    v108 = v17->m128i_u32[2];
    v111 = (__m128)a7;
LABEL_8:
    _RAX = v126;
    v19 = 64;
    __asm { tzcnt   rdx, rax }
    if ( v126 )
      v19 = _RDX;
    if ( v19 > v16 )
      v19 = v16;
    v21 = v19;
    goto LABEL_13;
  }
  sub_16A4FD0((__int64)&v126, (const void **)a3);
  v16 = v127;
  v70 = *(const __m128i **)(a2 + 32);
  a8 = _mm_loadu_si128(v70);
  v109 = v70->m128i_i64[0];
  v108 = v70->m128i_u32[2];
  v111 = (__m128)a8;
  if ( v127 <= 0x40 )
    goto LABEL_8;
  v21 = sub_16A58A0((__int64)&v126);
LABEL_13:
  if ( v21 )
  {
    v73 = 16LL * v108;
    v74 = sub_1E0A0C0(a4[4]);
    v75 = sub_1F40B60(
            a1,
            *(unsigned __int8 *)(v73 + *(_QWORD *)(v109 + 40)),
            *(_QWORD *)(v73 + *(_QWORD *)(v109 + 40) + 8),
            v74,
            1);
    *(_QWORD *)&v77 = sub_1D38BB0((__int64)a4, v21, (__int64)&v124, v75, v76, 0, a7, *(double *)a8.m128i_i64, a9, 0);
    v109 = (__int64)sub_1D332F0(
                      a4,
                      123,
                      (__int64)&v124,
                      *(unsigned __int8 *)(*(_QWORD *)(v109 + 40) + v73),
                      *(const void ***)(*(_QWORD *)(v109 + 40) + v73 + 8),
                      9u,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v111.m128_i64[0],
                      v111.m128_u64[1],
                      v77);
    v80 = *(unsigned int *)(a6 + 8);
    v108 = v81;
    if ( (unsigned int)v80 >= *(_DWORD *)(a6 + 12) )
    {
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v78, v79);
      v80 = *(unsigned int *)(a6 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a6 + 8 * v80) = v109;
    v16 = v127;
    ++*(_DWORD *)(a6 + 8);
    if ( v16 > 0x40 )
    {
      sub_16A5E70((__int64)&v126, v21);
      v16 = v127;
    }
    else
    {
      v82 = (__int64)(v126 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
      *(_QWORD *)&v82 = (__int64)v82 >> v21;
      if ( v21 == v16 )
        *(_QWORD *)&v82 = *((_QWORD *)&v82 + 1);
      v126 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & v82;
    }
  }
  v129 = 1;
  v128 = 0;
  v131 = v16;
  if ( v16 > 0x40 )
    sub_16A4FD0((__int64)&v130, (const void **)&v126);
  else
    v130 = v126;
  while ( 1 )
  {
    sub_16A7B50((__int64)&v134, (__int64)&v126, (__int64 *)&v130);
    if ( v129 > 0x40 && v128 )
      j_j___libc_free_0_0(v128);
    v22 = v135;
    v23 = (_QWORD *)v134;
    v135 = 0;
    v128 = v134;
    v129 = v22;
    if ( v22 > 0x40 )
    {
      if ( v22 - (unsigned int)sub_16A57B0((__int64)&v128) > 0x40 )
        goto LABEL_23;
      v23 = (_QWORD *)*v23;
    }
    if ( v23 == (_QWORD *)1 )
      break;
LABEL_23:
    v133 = v127;
    if ( v127 > 0x40 )
      sub_16A4EF0((__int64)&v132, 2, 0);
    else
      v132 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v127) & 2;
    sub_16A7590((__int64)&v132, (__int64 *)&v128);
    v24 = v133;
    v133 = 0;
    v135 = v24;
    v134 = v132;
    sub_16A7C10((__int64)&v130, (__int64 *)&v134);
    if ( v135 > 0x40 && v134 )
      j_j___libc_free_0_0(v134);
    if ( v133 > 0x40 )
    {
      if ( v132 )
        j_j___libc_free_0_0(v132);
    }
  }
  v65 = 16LL * v108;
  *(_QWORD *)&v66 = sub_1D38970(
                      (__int64)a4,
                      (__int64)&v130,
                      (__int64)&v124,
                      *(unsigned __int8 *)(v65 + *(_QWORD *)(v109 + 40)),
                      *(const void ***)(v65 + *(_QWORD *)(v109 + 40) + 8),
                      0,
                      a7,
                      *(double *)a8.m128i_i64,
                      a9,
                      0);
  v25 = sub_1D332F0(
          a4,
          54,
          (__int64)&v124,
          *(unsigned __int8 *)(*(_QWORD *)(v109 + 40) + v65),
          *(const void ***)(*(_QWORD *)(v109 + 40) + v65 + 8),
          0,
          *(double *)a7.m128i_i64,
          *(double *)a8.m128i_i64,
          a9,
          v109,
          v108 | v111.m128_u64[1] & 0xFFFFFFFF00000000LL,
          v66);
  v69 = *(unsigned int *)(a6 + 8);
  if ( (unsigned int)v69 >= *(_DWORD *)(a6 + 12) )
  {
    sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v67, v68);
    v69 = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v69) = v25;
  ++*(_DWORD *)(a6 + 8);
  if ( v131 > 0x40 && v130 )
    j_j___libc_free_0_0(v130);
  if ( v129 > 0x40 && v128 )
    j_j___libc_free_0_0(v128);
  if ( v127 > 0x40 )
  {
    v64 = v126;
    if ( v126 )
LABEL_52:
      j_j___libc_free_0_0(v64);
  }
LABEL_32:
  if ( v124 )
    sub_161E7C0((__int64)&v124, v124);
  return v25;
}
