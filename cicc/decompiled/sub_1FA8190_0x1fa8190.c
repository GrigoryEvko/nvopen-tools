// Function: sub_1FA8190
// Address: 0x1fa8190
//
__int64 __fastcall sub_1FA8190(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v7; // rax
  __m128 v8; // xmm0
  __m128i v9; // xmm1
  __int64 v10; // rbx
  __int64 v11; // r13
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // r9
  _QWORD *v15; // rdi
  __int64 v16; // r8
  unsigned int v17; // ecx
  _QWORD *v18; // rax
  unsigned int v19; // edx
  __int64 v20; // r14
  __int64 v21; // r9
  __int64 *v22; // rbx
  __int64 v23; // rax
  char v24; // r10
  __int64 v25; // r9
  __int64 v26; // r11
  __int64 v27; // rsi
  unsigned int v28; // r10d
  __int64 v29; // rcx
  __int64 *v30; // rax
  __int64 *v31; // r9
  unsigned int v32; // edx
  unsigned int v33; // ebx
  __int64 *v34; // r9
  __int64 v35; // r8
  __int16 v36; // ax
  unsigned int v37; // r14d
  __int128 v38; // rax
  __int128 v39; // kr00_16
  _QWORD *v40; // r14
  __int64 v41; // r11
  __int64 v42; // r10
  char v43; // si
  __int64 v44; // rbx
  unsigned int v45; // ecx
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rbx
  __int64 v49; // rsi
  __int64 *v50; // r14
  __int128 v51; // kr20_16
  __int64 *v52; // rax
  int v53; // edx
  int v54; // r14d
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdi
  __int64 i; // rbx
  _QWORD *v60; // rdi
  __int64 v61; // r8
  unsigned int v62; // ecx
  __int64 *v63; // rax
  unsigned int v64; // edx
  __int64 v65; // rdx
  __int64 v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // r9
  __int64 v70; // r10
  unsigned int v71; // ecx
  __int64 v72; // r11
  unsigned __int8 v73; // si
  __int64 v74; // rax
  __int64 v75; // rsi
  __int64 v76; // rdx
  __int64 v77; // rsi
  const void **v78; // r8
  __int64 v79; // rcx
  __int64 v80; // rax
  unsigned int v81; // edx
  __int64 v82; // [rsp+0h] [rbp-F0h]
  const void **v83; // [rsp+0h] [rbp-F0h]
  __int64 v84; // [rsp+8h] [rbp-E8h]
  unsigned int v85; // [rsp+10h] [rbp-E0h]
  __int64 v86; // [rsp+10h] [rbp-E0h]
  __int64 v87; // [rsp+18h] [rbp-D8h]
  __int64 v88; // [rsp+18h] [rbp-D8h]
  __int64 v89; // [rsp+18h] [rbp-D8h]
  __int64 v90; // [rsp+20h] [rbp-D0h]
  __int64 v91; // [rsp+20h] [rbp-D0h]
  __int64 v92; // [rsp+20h] [rbp-D0h]
  __int64 v93; // [rsp+20h] [rbp-D0h]
  unsigned int v94; // [rsp+28h] [rbp-C8h]
  unsigned int v95; // [rsp+28h] [rbp-C8h]
  char v96; // [rsp+28h] [rbp-C8h]
  char v97; // [rsp+28h] [rbp-C8h]
  __m128i v98; // [rsp+30h] [rbp-C0h]
  __int128 v99; // [rsp+30h] [rbp-C0h]
  unsigned __int16 v100; // [rsp+30h] [rbp-C0h]
  bool v101; // [rsp+40h] [rbp-B0h]
  unsigned int v102; // [rsp+40h] [rbp-B0h]
  __int64 v103; // [rsp+40h] [rbp-B0h]
  __int64 v104; // [rsp+40h] [rbp-B0h]
  __int128 v105; // [rsp+40h] [rbp-B0h]
  _QWORD *v106; // [rsp+40h] [rbp-B0h]
  int v107; // [rsp+50h] [rbp-A0h]
  __int128 v108; // [rsp+50h] [rbp-A0h]
  __int64 *v109; // [rsp+50h] [rbp-A0h]
  __int64 *v110; // [rsp+50h] [rbp-A0h]
  __int64 v111; // [rsp+50h] [rbp-A0h]
  __int64 v112; // [rsp+50h] [rbp-A0h]
  __int64 *v113; // [rsp+50h] [rbp-A0h]
  __int64 v114; // [rsp+50h] [rbp-A0h]
  __int64 v115; // [rsp+58h] [rbp-98h]
  unsigned int v116; // [rsp+60h] [rbp-90h]
  __int128 v117; // [rsp+60h] [rbp-90h]
  __int64 *v118; // [rsp+70h] [rbp-80h]
  __int64 v119; // [rsp+90h] [rbp-60h] BYREF
  int v120; // [rsp+98h] [rbp-58h]
  __m128i v121; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v122; // [rsp+B0h] [rbp-40h]
  __int64 v123; // [rsp+B8h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 32);
  v8 = (__m128)_mm_loadu_si128((const __m128i *)v7);
  v9 = _mm_loadu_si128((const __m128i *)(v7 + 5));
  v10 = *v7;
  v11 = v7[5];
  v12 = *((_DWORD *)v7 + 12);
  v107 = *((_DWORD *)v7 + 2);
  if ( (*(_BYTE *)(a2 + 26) & 8) == 0 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 40) + 16LL) == 1 )
    {
      if ( !(unsigned __int8)sub_1D18C40(a2, 0) )
      {
        v57 = *(_QWORD *)(*(_QWORD *)a1 + 664LL);
        v122 = *(_QWORD *)a1;
        v121.m128i_i64[1] = v57;
        *(_QWORD *)(v122 + 664) = &v121;
        v121.m128i_i64[0] = (__int64)off_49FFF30;
        v58 = *(_QWORD *)a1;
        v123 = a1;
        sub_1D44C70(v58, a2, 1, v8.m128_i64[0], v8.m128_u32[2]);
        for ( i = *(_QWORD *)(v10 + 48); i; i = *(_QWORD *)(i + 32) )
          sub_1F81BC0(a1, *(_QWORD *)(i + 16));
        if ( *(_QWORD *)(a2 + 48) )
          goto LABEL_53;
        goto LABEL_59;
      }
    }
    else
    {
      v13 = v7[10];
      v101 = *(_WORD *)(v13 + 24) == 32 && (*(_BYTE *)(v13 + 26) & 8) != 0;
      if ( !(unsigned __int8)sub_1D18C40(a2, 0) && (byte_4FCE780 == 1 && !v101 || !(unsigned __int8)sub_1D18C40(a2, 1)) )
      {
        v15 = *(_QWORD **)a1;
        v16 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
        v17 = **(unsigned __int8 **)(a2 + 40);
        v121.m128i_i64[0] = 0;
        v121.m128i_i32[2] = 0;
        v18 = sub_1D2B300(v15, 0x30u, (__int64)&v121, v17, v16, v14);
        v116 = v19;
        v20 = (__int64)v18;
        if ( v121.m128i_i64[0] )
          sub_161E7C0((__int64)&v121, v121.m128i_i64[0]);
        if ( (unsigned __int8)sub_1D18C40(a2, 1) && byte_4FCE780 == 1 && !v101 )
        {
          v22 = *(__int64 **)a1;
          v23 = *(_QWORD *)(a2 + 32);
          v24 = (*(_WORD *)(a2 + 26) >> 7) & 7;
          v25 = *(_QWORD *)(v23 + 80);
          v26 = *(_QWORD *)(v23 + 40);
          v98 = _mm_loadu_si128((const __m128i *)(v23 + 40));
          v102 = *(_DWORD *)(v23 + 48);
          v108 = (__int128)_mm_loadu_si128((const __m128i *)(v23 + 80));
          if ( *(_WORD *)(v25 + 24) == 32 )
          {
            v77 = *(_QWORD *)(v25 + 72);
            v78 = *(const void ***)(*(_QWORD *)(v25 + 40) + 8LL);
            v79 = **(unsigned __int8 **)(v25 + 40);
            v121.m128i_i64[0] = v77;
            if ( v77 )
            {
              v86 = v79;
              v83 = v78;
              v88 = v25;
              v92 = v26;
              v96 = v24;
              sub_1623A60((__int64)&v121, v77, 2);
              v79 = v86;
              v78 = v83;
              v25 = v88;
              v26 = v92;
              v24 = v96;
            }
            v93 = v26;
            v121.m128i_i32[2] = *(_DWORD *)(v25 + 64);
            v97 = v24;
            v80 = sub_1D37E40(
                    (__int64)v22,
                    *(_QWORD *)(v25 + 88),
                    (__int64)&v121,
                    v79,
                    v78,
                    0,
                    (__m128i)v8,
                    *(double *)v9.m128i_i64,
                    a5,
                    0);
            v25 = v80;
            v24 = v97;
            v26 = v93;
            *((_QWORD *)&v108 + 1) = v81 | *((_QWORD *)&v108 + 1) & 0xFFFFFFFF00000000LL;
            if ( v121.m128i_i64[0] )
            {
              v89 = v80;
              sub_161E7C0((__int64)&v121, v121.m128i_i64[0]);
              v22 = *(__int64 **)a1;
              v25 = v89;
              v26 = v93;
              v24 = v97;
            }
            else
            {
              v22 = *(__int64 **)a1;
            }
          }
          v27 = *(_QWORD *)(a2 + 72);
          v28 = ((v24 & 0xFD) != 1) + 52;
          v29 = *(unsigned __int8 *)(*(_QWORD *)(v26 + 40) + 16LL * v102);
          v121.m128i_i64[0] = v27;
          if ( v27 )
          {
            v90 = v29;
            v94 = v28;
            v103 = v25;
            sub_1623A60((__int64)&v121, v27, 2);
            v29 = v90;
            v28 = v94;
            v25 = v103;
          }
          *(_QWORD *)&v108 = v25;
          v121.m128i_i32[2] = *(_DWORD *)(a2 + 64);
          v30 = sub_1D332F0(
                  v22,
                  v28,
                  (__int64)&v121,
                  v29,
                  0,
                  0,
                  *(double *)v8.m128_u64,
                  *(double *)v9.m128i_i64,
                  a5,
                  v98.m128i_i64[0],
                  v98.m128i_u64[1],
                  v108);
          v31 = v30;
          v33 = v32;
          if ( v121.m128i_i64[0] )
          {
            v109 = v30;
            sub_161E7C0((__int64)&v121, v121.m128i_i64[0]);
            v31 = v109;
          }
          v110 = v31;
          sub_1FA7E80(a1, *(_QWORD *)(a2 + 48));
          v34 = v110;
        }
        else
        {
          v60 = *(_QWORD **)a1;
          v61 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 24LL);
          v62 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL);
          v121.m128i_i64[0] = 0;
          v121.m128i_i32[2] = 0;
          v63 = sub_1D2B300(v60, 0x30u, (__int64)&v121, v62, v61, v21);
          v34 = v63;
          v33 = v64;
          if ( v121.m128i_i64[0] )
          {
            v113 = v63;
            sub_161E7C0((__int64)&v121, v121.m128i_i64[0]);
            v34 = v113;
          }
        }
        v114 = (__int64)v34;
        v65 = *(_QWORD *)(*(_QWORD *)a1 + 664LL);
        v122 = *(_QWORD *)a1;
        v121.m128i_i64[1] = v65;
        *(_QWORD *)(v122 + 664) = &v121;
        v66 = *(_QWORD *)a1;
        v121.m128i_i64[0] = (__int64)off_49FFF30;
        v123 = a1;
        sub_1D44C70(v66, a2, 0, v20, v116);
        sub_1D44C70(*(_QWORD *)a1, a2, 1, v114, v33);
        sub_1D44C70(*(_QWORD *)a1, a2, 2, v8.m128_i64[0], v8.m128_u32[2]);
LABEL_59:
        sub_1F81E80((__int64 *)a1, a2);
LABEL_53:
        *(_QWORD *)(v122 + 664) = v121.m128i_i64[1];
        return a2;
      }
    }
  }
  v35 = *(unsigned int *)(a1 + 20);
  if ( (_DWORD)v35 )
  {
    v36 = (*(_WORD *)(a2 + 26) >> 7) & 7;
    if ( *(_WORD *)(a2 + 24) != 185 || (*(_BYTE *)(a2 + 27) & 0xC) != 0 )
    {
      if ( (_BYTE)v36 )
        goto LABEL_40;
    }
    else
    {
      if ( (_BYTE)v36 )
        goto LABEL_40;
      if ( (*(_BYTE *)(a2 + 26) & 8) == 0 && *(_WORD *)(v10 + 24) == 186 && (*(_BYTE *)(v10 + 27) & 4) == 0 )
      {
        v74 = *(_QWORD *)(v10 + 32);
        if ( *(_QWORD *)(v74 + 80) == v11 && *(_DWORD *)(v74 + 88) == v12 )
        {
          v75 = *(_QWORD *)(a2 + 40);
          v76 = *(_QWORD *)(*(_QWORD *)(v74 + 40) + 40LL) + 16LL * *(unsigned int *)(v74 + 48);
          if ( *(_BYTE *)v75 == *(_BYTE *)v76 && (*(_QWORD *)(v75 + 8) == *(_QWORD *)(v76 + 8) || *(_BYTE *)v76) )
            return sub_1F9A400(a1, a2, *(_QWORD *)(v74 + 40), *(_QWORD *)(v74 + 48), v8.m128_i64[0], v8.m128_i64[1], 1);
        }
      }
    }
    v37 = sub_1D1FC50(*(_QWORD *)a1, v9.m128i_i64[0]);
    if ( v37 )
    {
      if ( v37 > (unsigned int)sub_1E34390(*(_QWORD *)(a2 + 104)) )
      {
        v35 = *(_QWORD *)(a2 + 104);
        if ( !(*(_QWORD *)(v35 + 8) % (__int64)v37) )
        {
          v67 = *(_QWORD *)a1;
          a5 = _mm_loadu_si128((const __m128i *)(v35 + 40));
          v68 = *(_QWORD *)(a2 + 72);
          v121 = a5;
          v69 = *(unsigned __int8 *)(a2 + 88);
          v70 = *(_QWORD *)(a2 + 96);
          v106 = (_QWORD *)v67;
          v122 = *(_QWORD *)(v35 + 56);
          v100 = *(_WORD *)(v35 + 32);
          v71 = **(unsigned __int8 **)(a2 + 40);
          v72 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
          v119 = v68;
          if ( v68 )
          {
            v85 = v71;
            v82 = v69;
            v84 = v70;
            v87 = v72;
            v91 = v35;
            sub_1623A60((__int64)&v119, v68, 2);
            v71 = v85;
            v69 = v82;
            v70 = v84;
            v72 = v87;
            v35 = v91;
          }
          v73 = *(_BYTE *)(a2 + 27);
          v120 = *(_DWORD *)(a2 + 64);
          sub_1D2B810(
            v106,
            (v73 >> 2) & 3,
            (__int64)&v119,
            v71,
            v72,
            v37,
            *(_OWORD *)&v8,
            v9.m128i_i64[0],
            v9.m128i_i64[1],
            *(_OWORD *)v35,
            *(_QWORD *)(v35 + 16),
            v69,
            v70,
            v100,
            (__int64)&v121);
          if ( v119 )
            sub_161E7C0((__int64)&v119, v119);
        }
      }
    }
  }
  if ( (*(_WORD *)(a2 + 26) & 0x380) != 0
    || (*(_QWORD *)&v38 = sub_1F71D20(
                            (_QWORD *)a1,
                            a2,
                            (__int64 *)v8.m128_u64[0],
                            (__int64 *)v8.m128_u64[1],
                            (_DWORD *)v35,
                            v8,
                            *(double *)v9.m128i_i64,
                            a5),
        v39 = v38,
        v10 == (_QWORD)v38)
    && v107 == DWORD2(v38) )
  {
LABEL_40:
    if ( (unsigned __int8)sub_1F8F9B0((__int64 *)a1, a2, (__m128i)v8, *(double *)v9.m128i_i64, a5)
      || *(int *)(a1 + 16) > 2 && (unsigned __int8)sub_1F81F00((__int64 *)a1, a2) )
    {
      return a2;
    }
    else
    {
      return 0;
    }
  }
  v40 = *(_QWORD **)a1;
  v41 = *(_QWORD *)(a2 + 72);
  v42 = *(_QWORD *)(a2 + 104);
  v43 = (*(_BYTE *)(a2 + 27) >> 2) & 3;
  if ( v43 )
  {
    v111 = *(unsigned __int8 *)(a2 + 88);
    v115 = *(_QWORD *)(a2 + 96);
    v44 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
    v45 = **(unsigned __int8 **)(a2 + 40);
    v121.m128i_i64[0] = *(_QWORD *)(a2 + 72);
    if ( v41 )
    {
      v95 = v45;
      v99 = v38;
      v104 = v42;
      sub_1623A60((__int64)&v121, v41, 2);
      v45 = v95;
      v42 = v104;
      v43 = (*(_BYTE *)(a2 + 27) >> 2) & 3;
      v39 = v99;
    }
    v121.m128i_i32[2] = *(_DWORD *)(a2 + 64);
    v46 = sub_1D2B590(v40, v43, (__int64)&v121, v45, v44, v42, v39, v9.m128i_i64[0], v9.m128i_i64[1], v111, v115);
    v47 = v121.m128i_i64[0];
    v48 = v46;
    if ( !v121.m128i_i64[0] )
      goto LABEL_34;
  }
  else
  {
    v121.m128i_i64[0] = *(_QWORD *)(a2 + 72);
    if ( v41 )
    {
      v105 = v38;
      v112 = v42;
      sub_1623A60((__int64)&v121, v41, 2);
      v42 = v112;
      v39 = v105;
    }
    v121.m128i_i32[2] = *(_DWORD *)(a2 + 64);
    v56 = sub_1D2B660(
            v40,
            **(unsigned __int8 **)(a2 + 40),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
            (__int64)&v121,
            v39,
            *((__int64 *)&v39 + 1),
            v9.m128i_i64[0],
            v9.m128i_i64[1],
            v42);
    v47 = v121.m128i_i64[0];
    v48 = v56;
    if ( !v121.m128i_i64[0] )
      goto LABEL_34;
  }
  sub_161E7C0((__int64)&v121, v47);
LABEL_34:
  v49 = *(_QWORD *)(a2 + 72);
  v50 = *(__int64 **)a1;
  v121.m128i_i64[0] = v49;
  v51 = __PAIR128__(1, v48);
  if ( v49 )
  {
    *(_QWORD *)&v117 = v48;
    *((_QWORD *)&v117 + 1) = 1;
    sub_1623A60((__int64)&v121, v49, 2);
    v51 = v117;
  }
  v121.m128i_i32[2] = *(_DWORD *)(a2 + 64);
  v52 = sub_1D332F0(
          v50,
          2,
          (__int64)&v121,
          1,
          0,
          0,
          *(double *)v8.m128_u64,
          *(double *)v9.m128i_i64,
          a5,
          v8.m128_i64[0],
          v8.m128_u64[1],
          v51);
  v54 = v53;
  if ( v121.m128i_i64[0] )
  {
    v118 = v52;
    sub_161E7C0((__int64)&v121, v121.m128i_i64[0]);
    v52 = v118;
  }
  v121.m128i_i64[0] = v48;
  LODWORD(v123) = v54;
  v121.m128i_i32[2] = 0;
  v122 = (__int64)v52;
  return sub_1F994A0(a1, a2, v121.m128i_i64, 2, 1);
}
