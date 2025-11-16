// Function: sub_14285A0
// Address: 0x14285a0
//
__int64 __fastcall sub_14285A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v6; // rax
  __int64 v7; // r14
  int v8; // eax
  unsigned int v9; // edx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __m128i v12; // xmm7
  __m128i v13; // xmm5
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // r11
  __int64 v22; // rcx
  __int64 v23; // r10
  int v24; // r11d
  _QWORD *v25; // rax
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r9
  int v29; // eax
  int v30; // ebx
  bool v31; // zf
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  int v35; // eax
  unsigned __int32 v36; // r12d
  const __m128i *v37; // r13
  __int64 *v38; // rax
  int v39; // esi
  int v40; // edx
  unsigned int v41; // esi
  __m128i *v42; // r14
  int v43; // eax
  __int64 v44; // rdi
  unsigned __int32 *v45; // rbx
  __int64 v46; // r14
  const __m128i *v47; // r12
  int v48; // eax
  int v49; // eax
  unsigned __int64 v50; // rax
  unsigned __int32 *v51; // r12
  unsigned __int32 *v52; // rbx
  unsigned __int32 v53; // ecx
  __int64 v54; // rdx
  __int64 *m128i_i64; // rdx
  __int64 v56; // rcx
  unsigned __int64 v57; // rax
  __int64 v58; // rdx
  char *v59; // rdi
  int v60; // eax
  unsigned int v61; // ecx
  __int64 v62; // rdx
  _QWORD *v63; // rax
  _QWORD *i; // rdx
  __int64 v66; // rax
  __int64 v67; // r12
  char *v68; // rbx
  const __m128i *v69; // r13
  __int64 v70; // rax
  __int64 v71; // rax
  char *m128i_i8; // rax
  __int64 v73; // rax
  int v74; // edx
  int v75; // r9d
  char v76; // dl
  char v77; // al
  __m128i *v78; // r13
  int v79; // edx
  __int64 v80; // rax
  char *v81; // r12
  char *v82; // rbx
  __int64 v83; // rdx
  __int32 v84; // eax
  __int64 v85; // rax
  __int64 v86; // rcx
  __m128i *v87; // r13
  __int64 v88; // rdx
  __int64 *v89; // r12
  __int64 *v90; // rbx
  __int64 *v91; // rdi
  __int64 v92; // rdx
  __int32 v93; // eax
  __int64 v94; // rdx
  __int64 v95; // rcx
  _QWORD *v96; // rdi
  unsigned int v97; // eax
  __int64 v98; // rax
  unsigned __int64 v99; // rax
  __int64 v100; // rax
  int v101; // r13d
  __int64 v102; // r12
  _QWORD *v103; // rax
  __int64 v104; // rdx
  _QWORD *j; // rdx
  _QWORD *v106; // rax
  unsigned __int64 v107; // [rsp+18h] [rbp-228h]
  __int64 v108; // [rsp+30h] [rbp-210h]
  unsigned int *v109; // [rsp+38h] [rbp-208h]
  __int64 v110; // [rsp+40h] [rbp-200h]
  __int64 v111; // [rsp+40h] [rbp-200h]
  __m128i *v112; // [rsp+48h] [rbp-1F8h]
  __int64 v113; // [rsp+48h] [rbp-1F8h]
  unsigned __int32 *v114; // [rsp+50h] [rbp-1F0h] BYREF
  __int64 v115; // [rsp+58h] [rbp-1E8h]
  _BYTE v116[32]; // [rsp+60h] [rbp-1E0h] BYREF
  __m128i v117; // [rsp+80h] [rbp-1C0h] BYREF
  __m128i v118; // [rsp+90h] [rbp-1B0h]
  __int64 v119; // [rsp+A0h] [rbp-1A0h]
  __int64 v120; // [rsp+A8h] [rbp-198h]
  __m128i *v121; // [rsp+B0h] [rbp-190h]
  __int64 v122; // [rsp+B8h] [rbp-188h]
  _BYTE *v123; // [rsp+C0h] [rbp-180h] BYREF
  __int64 v124; // [rsp+C8h] [rbp-178h]
  _BYTE v125[64]; // [rsp+D0h] [rbp-170h] BYREF
  __m128i *v126; // [rsp+110h] [rbp-130h] BYREF
  __int64 v127; // [rsp+118h] [rbp-128h]
  _BYTE v128[64]; // [rsp+120h] [rbp-120h] BYREF
  __m128i *v129; // [rsp+160h] [rbp-E0h] BYREF
  __int64 v130; // [rsp+168h] [rbp-D8h]
  _BYTE v131[64]; // [rsp+170h] [rbp-D0h] BYREF
  __int64 v132; // [rsp+1B0h] [rbp-90h] BYREF
  __m128i v133; // [rsp+1B8h] [rbp-88h] BYREF
  __m128i v134; // [rsp+1C8h] [rbp-78h] BYREF
  __int64 v135; // [rsp+1D8h] [rbp-68h]

  v4 = a2;
  *(_QWORD *)(a1 + 24) = a3;
  if ( *(_BYTE *)(a2 + 16) == 21 )
  {
    v4 = *(_QWORD *)(a2 - 24);
    v122 = 0;
    v6 = *(_QWORD *)(a3 + 40);
    v120 = v4;
    v119 = v6;
    v121 = (__m128i *)v4;
    v117 = _mm_loadu_si128((const __m128i *)(a3 + 8));
    v118 = _mm_loadu_si128((const __m128i *)(a3 + 24));
    if ( !v4 )
    {
      v112 = 0;
      goto LABEL_11;
    }
  }
  else
  {
    v120 = a2;
    v122 = 0;
    v73 = *(_QWORD *)(a3 + 40);
    v117 = _mm_loadu_si128((const __m128i *)(a3 + 8));
    v119 = v73;
    v118 = _mm_loadu_si128((const __m128i *)(a3 + 24));
  }
  v7 = v4;
  do
  {
    v121 = (__m128i *)v7;
    v8 = *(unsigned __int8 *)(v7 + 16);
    if ( (_BYTE)v8 == 22 )
    {
      if ( *(_QWORD *)(*(_QWORD *)a1 + 120LL) == v7 )
      {
        v76 = *(_BYTE *)(a3 + 65);
        v77 = 3;
      }
      else
      {
        sub_14203A0((bool *)&v132, v7, &v117, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 48LL), *(_QWORD **)(a1 + 8));
        if ( !(_BYTE)v132 )
        {
          v8 = *(unsigned __int8 *)(v7 + 16);
          goto LABEL_9;
        }
        if ( !BYTE2(v132) )
        {
          if ( *(_BYTE *)(a3 + 65) )
            *(_BYTE *)(a3 + 65) = 0;
          return v7;
        }
        v77 = BYTE1(v132);
        v76 = *(_BYTE *)(a3 + 65);
      }
      *(_BYTE *)(a3 + 64) = v77;
      if ( !v76 )
        *(_BYTE *)(a3 + 65) = 1;
      return v7;
    }
LABEL_9:
    if ( (unsigned int)(v8 - 21) > 1 )
      break;
    v7 = *(_QWORD *)(v7 - 24);
  }
  while ( v7 );
  v112 = v121;
LABEL_11:
  v9 = *(_DWORD *)(a1 + 40);
  if ( v9 >= *(_DWORD *)(a1 + 44) )
  {
    sub_1420280(a1 + 32);
    v9 = *(_DWORD *)(a1 + 40);
  }
  v10 = *(_QWORD *)(a1 + 32) + ((unsigned __int64)v9 << 6);
  if ( v10 )
  {
    v11 = *(_QWORD *)(a3 + 40);
    v12 = _mm_loadu_si128((const __m128i *)(a3 + 8));
    *(_QWORD *)(v10 + 40) = v4;
    v13 = _mm_loadu_si128((const __m128i *)(a3 + 24));
    *(_BYTE *)(v10 + 60) = 0;
    *(_QWORD *)(v10 + 32) = v11;
    *(_QWORD *)(v10 + 48) = v112;
    *(__m128i *)v10 = v12;
    *(__m128i *)(v10 + 16) = v13;
    v9 = *(_DWORD *)(a1 + 40);
  }
  *(_DWORD *)(a1 + 40) = v9 + 1;
  v107 = v9 + 1;
  v123 = v125;
  v124 = 0x1000000000LL;
  v114 = (unsigned __int32 *)v116;
  v115 = 0x800000000LL;
  v126 = (__m128i *)v128;
  v127 = 0x400000000LL;
  sub_1420A50(a1, v112, (__int64)&v123, 0);
  while ( 2 )
  {
    v14 = *(_QWORD *)a1;
    v15 = *(_QWORD *)(a1 + 16);
    v108 = *(_QWORD *)(*(_QWORD *)a1 + 120LL);
    v16 = *(unsigned int *)(v15 + 48);
    if ( !(_DWORD)v16 )
LABEL_181:
      BUG();
    v17 = v112[4].m128i_i64[0];
    v18 = *(_QWORD *)(v15 + 32);
    v19 = (v16 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v20 = (__int64 *)(v18 + 16LL * v19);
    v21 = *v20;
    if ( v17 != *v20 )
    {
      v74 = 1;
      while ( v21 != -8 )
      {
        v75 = v74 + 1;
        v19 = (v16 - 1) & (v74 + v19);
        v20 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v17 == *v20 )
          goto LABEL_18;
        v74 = v75;
      }
      goto LABEL_181;
    }
LABEL_18:
    if ( v20 == (__int64 *)(v18 + 16 * v16) )
      goto LABEL_181;
    v22 = *(_QWORD *)(v20[1] + 8);
    if ( !v22 )
      goto LABEL_33;
    v23 = *(_QWORD *)(v14 + 96);
    v17 = *(unsigned int *)(v14 + 112);
    v24 = v17 - 1;
    while ( 2 )
    {
      if ( !(_DWORD)v17 )
      {
LABEL_23:
        v22 = *(_QWORD *)(v22 + 8);
        if ( !v22 )
          goto LABEL_33;
        continue;
      }
      break;
    }
    v26 = v24 & (((unsigned int)*(_QWORD *)v22 >> 9) ^ ((unsigned int)*(_QWORD *)v22 >> 4));
    v27 = (__int64 *)(v23 + 16LL * v26);
    v28 = *v27;
    if ( *(_QWORD *)v22 != *v27 )
    {
      v29 = 1;
      while ( v28 != -8 )
      {
        v30 = v29 + 1;
        v26 = v24 & (v29 + v26);
        v27 = (__int64 *)(v23 + 16LL * v26);
        v28 = *v27;
        if ( *(_QWORD *)v22 == *v27 )
          goto LABEL_21;
        v29 = v30;
      }
      goto LABEL_23;
    }
LABEL_21:
    if ( (__int64 *)(v23 + 16 * v17) == v27 )
      goto LABEL_23;
    v25 = (_QWORD *)v27[1];
    if ( !v25 )
      goto LABEL_23;
    v32 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
    v31 = v32 == 0;
    v33 = v32 - 48;
    v34 = 0;
    if ( !v31 )
      v34 = v33;
    v108 = v34;
LABEL_33:
    v35 = v124;
    if ( (_DWORD)v124 )
    {
      v110 = a1 + 2096;
      while ( 1 )
      {
        v36 = *(_DWORD *)&v123[4 * v35 - 4];
        LODWORD(v124) = v35 - 1;
        v113 = v36;
        v37 = (const __m128i *)(*(_QWORD *)(a1 + 32) + ((unsigned __int64)v36 << 6));
        v132 = v37[3].m128i_i64[0];
        v133 = _mm_loadu_si128(v37);
        v134 = _mm_loadu_si128(v37 + 1);
        v135 = v37[2].m128i_i64[0];
        v17 = (unsigned int)sub_14244A0(v110, &v132, (__int64 **)&v129);
        v38 = (__int64 *)v129;
        if ( (_BYTE)v17 )
          goto LABEL_35;
        v39 = *(_DWORD *)(a1 + 2112);
        ++*(_QWORD *)(a1 + 2096);
        v40 = v39 + 1;
        v41 = *(_DWORD *)(a1 + 2120);
        v17 = 2 * v41;
        if ( 4 * v40 >= 3 * v41 )
        {
          v41 *= 2;
        }
        else
        {
          v22 = v41 - *(_DWORD *)(a1 + 2116) - v40;
          if ( (unsigned int)v22 > v41 >> 3 )
            goto LABEL_39;
        }
        sub_1425980(v110, v41);
        sub_14244A0(v110, &v132, (__int64 **)&v129);
        v38 = (__int64 *)v129;
        v40 = *(_DWORD *)(a1 + 2112) + 1;
LABEL_39:
        *(_DWORD *)(a1 + 2112) = v40;
        if ( *v38 != -8 || v38[1] != -8 || v38[2] || v38[3] || v38[4] || v38[5] )
          --*(_DWORD *)(a1 + 2116);
        *v38 = v132;
        *(__m128i *)(v38 + 1) = _mm_loadu_si128(&v133);
        *(__m128i *)(v38 + 3) = _mm_loadu_si128(&v134);
        v38[5] = v135;
        v42 = (__m128i *)v37[3].m128i_i64[0];
        if ( v42 )
        {
          while ( 1 )
          {
            v37[3].m128i_i64[0] = (__int64)v42;
            if ( (__m128i *)v108 == v42 )
              goto LABEL_102;
            v43 = v42[1].m128i_u8[0];
            if ( (_BYTE)v43 != 22 )
              goto LABEL_49;
            v44 = *(_QWORD *)a1;
            if ( *(__m128i **)(*(_QWORD *)a1 + 120LL) == v42 )
              goto LABEL_75;
            sub_14203A0(
              (bool *)&v132,
              (__int64)v42,
              v37,
              *(_QWORD *)(*(_QWORD *)(a1 + 24) + 48LL),
              *(_QWORD **)(a1 + 8));
            if ( (_BYTE)v132 )
              break;
            v43 = v42[1].m128i_u8[0];
LABEL_49:
            if ( (unsigned int)(v43 - 21) <= 1 )
            {
              v42 = (__m128i *)v42[-2].m128i_i64[1];
              if ( v42 )
                continue;
            }
            v42 = (__m128i *)v37[3].m128i_i64[0];
            goto LABEL_51;
          }
          v44 = *(_QWORD *)a1;
LABEL_75:
          if ( !sub_1428550(v44, (__int64)v42, v108, v22) )
          {
            while ( 1 )
            {
              v56 = *(_QWORD *)(a1 + 32) + (v113 << 6);
              v57 = v113 << 6 >> 6;
              if ( v57 < v107 )
                goto LABEL_77;
              if ( !*(_BYTE *)(v56 + 60) )
                break;
              v113 = *(unsigned int *)(v56 + 56);
            }
            v56 = *(_QWORD *)(a1 + 32);
            LODWORD(v57) = 0;
LABEL_77:
            v58 = *(_QWORD *)(v56 + 48);
            v133.m128i_i32[0] = v57;
            v133.m128i_i64[1] = (__int64)&v134.m128i_i64[1];
            v59 = (char *)v126;
            v132 = v58;
            v134.m128i_i64[0] = 0x400000000LL;
LABEL_78:
            if ( v59 == v128 )
              goto LABEL_80;
LABEL_79:
            _libc_free((unsigned __int64)v59);
            goto LABEL_80;
          }
          v71 = (unsigned int)v127;
          if ( (unsigned int)v127 >= HIDWORD(v127) )
          {
            sub_16CD150(&v126, v128, 0, 16);
            v71 = (unsigned int)v127;
          }
          m128i_i8 = v126[v71].m128i_i8;
          *(_QWORD *)m128i_i8 = v42;
          *((_QWORD *)m128i_i8 + 1) = v36;
          LODWORD(v127) = v127 + 1;
LABEL_35:
          v35 = v124;
          if ( !(_DWORD)v124 )
            break;
        }
        else
        {
LABEL_51:
          if ( (__m128i *)v108 == v42 )
          {
LABEL_102:
            v66 = (unsigned int)v115;
            if ( (unsigned int)v115 >= HIDWORD(v115) )
            {
              sub_16CD150(&v114, v116, 0, 4);
              v66 = (unsigned int)v115;
            }
            v114[v66] = v36;
            LODWORD(v115) = v115 + 1;
            goto LABEL_35;
          }
          sub_1420A50(a1, v42, (__int64)&v123, v36);
          v35 = v124;
          if ( !(_DWORD)v124 )
            break;
        }
      }
    }
    if ( !(_DWORD)v115 )
    {
      v78 = v126;
      v79 = v127;
      v80 = (unsigned int)v127;
      v81 = v126[1].m128i_i8;
      v82 = v126[v80].m128i_i8;
      if ( &v126[v80] == &v126[1] )
      {
        v59 = (char *)v126;
      }
      else
      {
        do
        {
          if ( !sub_1428550(*(_QWORD *)a1, *(_QWORD *)v81, v78->m128i_i64[0], v22) )
            v78 = (__m128i *)v81;
          v81 += 16;
        }
        while ( v82 != v81 );
        v59 = (char *)v126;
        v79 = v127;
        v80 = (unsigned int)v127;
        v81 = v126[v80].m128i_i8;
      }
      if ( v78 != (__m128i *)(v81 - 16) )
      {
        v83 = *((_QWORD *)v81 - 2);
        v84 = *((_DWORD *)v81 - 2);
        *((__m128i *)v81 - 1) = _mm_loadu_si128(v78);
        v78->m128i_i64[0] = v83;
        v59 = (char *)v126;
        v78->m128i_i32[2] = v84;
        v79 = v127;
        v80 = (unsigned int)v127;
      }
      v85 = (__int64)&v59[v80 * 16 - 16];
      v86 = *(_QWORD *)v85;
      LODWORD(v85) = *(_DWORD *)(v85 + 8);
      LODWORD(v127) = v79 - 1;
      v133.m128i_i64[1] = (__int64)&v134.m128i_i64[1];
      v133.m128i_i32[0] = v85;
      v132 = v86;
      v134.m128i_i64[0] = 0x400000000LL;
      if ( v79 != 1 )
      {
        sub_14200F0((__int64)&v133.m128i_i64[1], (char **)&v126);
        v59 = (char *)v126;
      }
      goto LABEL_78;
    }
    v45 = v114;
    v112 = 0;
    v129 = (__m128i *)v131;
    v130 = 0x400000000LL;
    v109 = &v114[(unsigned int)v115];
    while ( 2 )
    {
      v111 = *v45;
      v46 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + (v111 << 6) + 48);
      v47 = (const __m128i *)(*(_QWORD *)(a1 + 32) + (v111 << 6));
      if ( !v46 )
      {
        v112 = 0;
        goto LABEL_64;
      }
      while ( 2 )
      {
        v47[3].m128i_i64[0] = v46;
        v48 = *(unsigned __int8 *)(v46 + 16);
        if ( (_BYTE)v48 != 22 )
        {
LABEL_62:
          if ( (unsigned int)(v48 - 21) > 1 || (v46 = *(_QWORD *)(v46 - 24)) == 0 )
          {
            v112 = (__m128i *)v47[3].m128i_i64[0];
            goto LABEL_64;
          }
          continue;
        }
        break;
      }
      if ( *(_QWORD *)(*(_QWORD *)a1 + 120LL) != v46 )
      {
        sub_14203A0((bool *)&v132, v46, v47, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 48LL), *(_QWORD **)(a1 + 8));
        if ( !(_BYTE)v132 )
        {
          v48 = *(unsigned __int8 *)(v46 + 16);
          goto LABEL_62;
        }
      }
      v54 = (unsigned int)v130;
      if ( (unsigned int)v130 >= HIDWORD(v130) )
      {
        sub_16CD150(&v129, v131, 0, 16);
        v54 = (unsigned int)v130;
      }
      m128i_i64 = v129[v54].m128i_i64;
      *m128i_i64 = v46;
      m128i_i64[1] = v111;
      LODWORD(v130) = v130 + 1;
LABEL_64:
      if ( v109 != ++v45 )
        continue;
      break;
    }
    if ( (_DWORD)v127 )
    {
      if ( !v112 )
      {
        v22 = v108;
        if ( v108 )
        {
          while ( (unsigned int)*(unsigned __int8 *)(v22 + 16) - 21 <= 1 && *(_QWORD *)(v22 - 24) )
            v22 = *(_QWORD *)(v22 - 24);
          v108 = v22;
        }
        v112 = (__m128i *)v108;
      }
      v67 = v112[4].m128i_i64[0];
      v68 = v126[(unsigned int)v127].m128i_i8;
      v69 = v126;
      do
      {
        if ( (unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 16), v67, *(_QWORD *)(v69->m128i_i64[0] + 64), v22, v17) )
        {
          v70 = (unsigned int)v130;
          if ( (unsigned int)v130 >= HIDWORD(v130) )
          {
            sub_16CD150(&v129, v131, 0, 16);
            v70 = (unsigned int)v130;
          }
          v129[v70] = _mm_loadu_si128(v69);
          LODWORD(v130) = v130 + 1;
        }
        ++v69;
      }
      while ( v68 != (char *)v69 );
    }
    v49 = v130;
    if ( !(_DWORD)v130 )
    {
      v50 = *(unsigned int *)(a1 + 40);
      v51 = v114;
      LODWORD(v124) = 0;
      v107 = v50;
      v52 = &v114[(unsigned int)v115];
      if ( v114 != v52 )
      {
        do
        {
          v53 = *v51++;
          sub_1420A50(a1, v112, (__int64)&v123, v53);
        }
        while ( v52 != v51 );
      }
      LODWORD(v115) = 0;
      if ( v129 != (__m128i *)v131 )
        _libc_free((unsigned __int64)v129);
      continue;
    }
    break;
  }
  v87 = v129;
  v88 = 2LL * (unsigned int)v130;
  v89 = v129[1].m128i_i64;
  v90 = v129[(unsigned __int64)v88 / 2].m128i_i64;
  if ( &v129[(unsigned __int64)v88 / 2] == &v129[1] )
  {
    v91 = (__int64 *)v129;
  }
  else
  {
    do
    {
      if ( !sub_1428550(*(_QWORD *)a1, *v89, v87->m128i_i64[0], v22) )
        v87 = (__m128i *)v89;
      v89 += 2;
    }
    while ( v90 != v89 );
    v91 = (__int64 *)v129;
    v49 = v130;
    v88 = 2LL * (unsigned int)v130;
    v89 = v129[(unsigned __int64)v88 / 2].m128i_i64;
  }
  if ( v87 != (__m128i *)(v89 - 2) )
  {
    v92 = *(v89 - 2);
    v93 = *((_DWORD *)v89 - 2);
    *((__m128i *)v89 - 1) = _mm_loadu_si128(v87);
    v87->m128i_i32[2] = v93;
    v87->m128i_i64[0] = v92;
    v91 = (__int64 *)v129;
    v49 = v130;
    v88 = 2LL * (unsigned int)v130;
  }
  v94 = (__int64)&v91[v88 - 2];
  v95 = *(_QWORD *)v94;
  LODWORD(v94) = *(_DWORD *)(v94 + 8);
  LODWORD(v130) = v49 - 1;
  v133.m128i_i64[1] = (__int64)&v134.m128i_i64[1];
  v132 = v95;
  v133.m128i_i32[0] = v94;
  v134.m128i_i64[0] = 0x400000000LL;
  if ( v49 != 1 )
  {
    sub_14200F0((__int64)&v133.m128i_i64[1], (char **)&v129);
    v91 = (__int64 *)v129;
  }
  if ( v91 != (__int64 *)v131 )
    _libc_free((unsigned __int64)v91);
  v59 = (char *)v126;
  if ( v126 != (__m128i *)v128 )
    goto LABEL_79;
LABEL_80:
  if ( v114 != (unsigned __int32 *)v116 )
    _libc_free((unsigned __int64)v114);
  if ( v123 != v125 )
    _libc_free((unsigned __int64)v123);
  v60 = *(_DWORD *)(a1 + 2112);
  ++*(_QWORD *)(a1 + 2096);
  *(_DWORD *)(a1 + 40) = 0;
  if ( v60 )
  {
    v61 = 4 * v60;
    v62 = *(unsigned int *)(a1 + 2120);
    if ( (unsigned int)(4 * v60) < 0x40 )
      v61 = 64;
    if ( v61 >= (unsigned int)v62 )
    {
LABEL_88:
      v63 = *(_QWORD **)(a1 + 2104);
      for ( i = &v63[6 * v62]; i != v63; *(v63 - 1) = 0 )
      {
        *v63 = -8;
        v63 += 6;
        *(v63 - 5) = -8;
        *(v63 - 4) = 0;
        *(v63 - 3) = 0;
        *(v63 - 2) = 0;
      }
      *(_QWORD *)(a1 + 2112) = 0;
      goto LABEL_91;
    }
    v96 = *(_QWORD **)(a1 + 2104);
    v97 = v60 - 1;
    if ( !v97 )
    {
      v102 = 6144;
      v101 = 128;
      goto LABEL_170;
    }
    _BitScanReverse(&v97, v97);
    v98 = (unsigned int)(1 << (33 - (v97 ^ 0x1F)));
    if ( (int)v98 < 64 )
      v98 = 64;
    if ( (_DWORD)v98 == (_DWORD)v62 )
    {
      *(_QWORD *)(a1 + 2112) = 0;
      v106 = &v96[6 * v98];
      do
      {
        if ( v96 )
        {
          *v96 = -8;
          v96[1] = -8;
          v96[2] = 0;
          v96[3] = 0;
          v96[4] = 0;
          v96[5] = 0;
        }
        v96 += 6;
      }
      while ( v106 != v96 );
    }
    else
    {
      v99 = ((unsigned __int64)(4 * (int)v98 / 3u + 1) >> 1)
          | (4 * (int)v98 / 3u + 1)
          | ((((unsigned __int64)(4 * (int)v98 / 3u + 1) >> 1) | (4 * (int)v98 / 3u + 1)) >> 2);
      v100 = (((v99 | (v99 >> 4)) >> 8) | v99 | (v99 >> 4) | ((((v99 | (v99 >> 4)) >> 8) | v99 | (v99 >> 4)) >> 16)) + 1;
      v101 = v100;
      v102 = 48 * v100;
LABEL_170:
      j___libc_free_0(v96);
      *(_DWORD *)(a1 + 2120) = v101;
      v103 = (_QWORD *)sub_22077B0(v102);
      v104 = *(unsigned int *)(a1 + 2120);
      *(_QWORD *)(a1 + 2112) = 0;
      *(_QWORD *)(a1 + 2104) = v103;
      for ( j = &v103[6 * v104]; j != v103; v103 += 6 )
      {
        if ( v103 )
        {
          *v103 = -8;
          v103[1] = -8;
          v103[2] = 0;
          v103[3] = 0;
          v103[4] = 0;
          v103[5] = 0;
        }
      }
    }
  }
  else if ( *(_DWORD *)(a1 + 2116) )
  {
    v62 = *(unsigned int *)(a1 + 2120);
    if ( (unsigned int)v62 <= 0x40 )
      goto LABEL_88;
    j___libc_free_0(*(_QWORD *)(a1 + 2104));
    *(_QWORD *)(a1 + 2104) = 0;
    *(_QWORD *)(a1 + 2112) = 0;
    *(_DWORD *)(a1 + 2120) = 0;
  }
LABEL_91:
  v7 = v132;
  if ( (unsigned __int64 *)v133.m128i_i64[1] != &v134.m128i_u64[1] )
    _libc_free(v133.m128i_u64[1]);
  return v7;
}
