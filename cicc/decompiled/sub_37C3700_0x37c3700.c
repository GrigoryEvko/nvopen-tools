// Function: sub_37C3700
// Address: 0x37c3700
//
__int64 __fastcall sub_37C3700(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i *a5, __int64 a6)
{
  unsigned __int64 v6; // rbx
  const void *v7; // r14
  __int64 v8; // r12
  __int64 *v9; // r13
  __int64 v10; // r12
  __int64 v11; // rbx
  unsigned __int64 v12; // rax
  __int64 *v13; // r13
  __int64 v14; // rax
  unsigned int v15; // r14d
  unsigned int v16; // esi
  __int64 v17; // rcx
  int v18; // r11d
  __int64 v19; // r8
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // r10
  __int64 *v23; // r8
  __int64 *v24; // r12
  __int64 v25; // rbx
  __int64 v26; // rsi
  bool v27; // zf
  __int64 *v28; // rdx
  __int64 *v29; // rcx
  __int64 v30; // rax
  char v31; // di
  __int64 v32; // r10
  int v33; // esi
  unsigned int v34; // ecx
  __int64 *v35; // rdx
  __int64 v36; // r11
  __int64 v37; // r15
  bool v38; // cf
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // r11
  unsigned __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // rbx
  unsigned int v47; // r9d
  const __m128i *v48; // r15
  __int64 v49; // r14
  __int64 i; // rax
  unsigned __int8 v51; // r13
  __int64 v52; // r15
  int v53; // eax
  unsigned int v54; // r12d
  int v55; // eax
  char v56; // dl
  int v57; // ecx
  char v58; // al
  __int64 v60; // rsi
  __int64 v61; // rsi
  int v62; // edx
  __int64 *v63; // rdi
  char v64; // al
  int v65; // r8d
  int v66; // eax
  int v67; // edx
  __int64 v68; // rcx
  unsigned __int8 v69; // r9
  __int64 v70; // r15
  int v71; // r14d
  __int64 v72; // r13
  __int64 v73; // r13
  __int64 v74; // rbx
  __int64 v75; // r15
  char v76; // r12
  _DWORD *v77; // r14
  int v78; // eax
  __int64 v79; // rdx
  __int64 v80; // rsi
  _DWORD *v81; // rdx
  _DWORD *v82; // rax
  _DWORD *v83; // rsi
  int v84; // esi
  __int32 v85; // ecx
  __m128i *v86; // rax
  __m128i v87; // xmm6
  unsigned int v88; // eax
  __m128i v89; // xmm7
  __m128i v90; // xmm3
  __m128i si128; // xmm4
  __int64 v92; // rax
  char v93; // al
  char v94; // al
  __int64 v98; // [rsp+10h] [rbp-190h]
  __int64 *v100; // [rsp+28h] [rbp-178h]
  __int64 v102; // [rsp+30h] [rbp-170h]
  __int64 *v103; // [rsp+38h] [rbp-168h]
  unsigned int v104; // [rsp+38h] [rbp-168h]
  __int64 v105; // [rsp+38h] [rbp-168h]
  __int64 *v106; // [rsp+40h] [rbp-160h]
  __int64 *v107; // [rsp+40h] [rbp-160h]
  int v108; // [rsp+40h] [rbp-160h]
  _BYTE *v109; // [rsp+40h] [rbp-160h]
  __int64 v110; // [rsp+48h] [rbp-158h]
  int v111; // [rsp+48h] [rbp-158h]
  _BYTE *v112; // [rsp+48h] [rbp-158h]
  unsigned __int8 v113; // [rsp+48h] [rbp-158h]
  unsigned __int8 v114; // [rsp+48h] [rbp-158h]
  char v115; // [rsp+48h] [rbp-158h]
  unsigned __int8 v116; // [rsp+48h] [rbp-158h]
  __m128i v117; // [rsp+50h] [rbp-150h] BYREF
  __m128i v118; // [rsp+60h] [rbp-140h] BYREF
  _DWORD v119[2]; // [rsp+70h] [rbp-130h] BYREF
  __m128i v120; // [rsp+78h] [rbp-128h]
  int v121; // [rsp+88h] [rbp-118h]
  __int64 *v122; // [rsp+90h] [rbp-110h] BYREF
  __int64 v123; // [rsp+98h] [rbp-108h]
  _BYTE v124[64]; // [rsp+A0h] [rbp-100h] BYREF
  _BYTE *v125; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v126; // [rsp+E8h] [rbp-B8h]
  _BYTE v127[176]; // [rsp+F0h] [rbp-B0h] BYREF

  v6 = *(unsigned int *)(a2 + 72);
  v122 = (__int64 *)v124;
  v7 = *(const void **)(a2 + 64);
  v8 = 8 * v6;
  v123 = 0x800000000LL;
  if ( v6 > 8 )
  {
    sub_C8D5F0((__int64)&v122, v124, v6, 8u, (__int64)a5, a6);
    v63 = &v122[(unsigned int)v123];
  }
  else
  {
    v9 = (__int64 *)v124;
    if ( !v8 )
      goto LABEL_3;
    v63 = (__int64 *)v124;
  }
  memcpy(v63, v7, 8 * v6);
  v9 = v122;
  LODWORD(v8) = v123;
LABEL_3:
  LODWORD(v123) = v8 + v6;
  v10 = (unsigned int)(v8 + v6);
  v11 = a1 + 664;
  v100 = &v9[v10];
  if ( v9 != &v9[v10] )
  {
    _BitScanReverse64(&v12, (v10 * 8) >> 3);
    sub_37C29D0(v9, (char *)&v9[v10], 2LL * (int)(63 - (v12 ^ 0x3F)), a1);
    if ( (unsigned __int64)v10 <= 16 )
    {
      sub_37C0740(v9, v100, a1);
    }
    else
    {
      v103 = v9 + 16;
      sub_37C0740(v9, v9 + 16, a1);
      if ( &v9[v10] != v9 + 16 )
      {
        do
        {
          v13 = v103;
          v110 = *v103;
          while ( 1 )
          {
            v14 = *(v13 - 1);
            v106 = v13--;
            v117.m128i_i64[0] = v110;
            v125 = (_BYTE *)v14;
            v15 = *(_DWORD *)sub_2E51790(v11, v117.m128i_i64);
            if ( v15 >= *(_DWORD *)sub_2E51790(v11, (__int64 *)&v125) )
              break;
            v13[1] = *v13;
          }
          ++v103;
          *v106 = v110;
        }
        while ( v100 != v103 );
      }
    }
  }
  v16 = *(_DWORD *)(a1 + 688);
  v117.m128i_i64[0] = a2;
  if ( !v16 )
  {
    v125 = 0;
    ++*(_QWORD *)(a1 + 664);
    goto LABEL_95;
  }
  v17 = *(_QWORD *)(a1 + 672);
  v18 = 1;
  v19 = 0;
  v20 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v21 = v17 + 16LL * v20;
  v22 = *(_QWORD *)v21;
  if ( a2 != *(_QWORD *)v21 )
  {
    while ( v22 != -4096 )
    {
      if ( v22 == -8192 && !v19 )
        v19 = v21;
      v20 = (v16 - 1) & (v18 + v20);
      v21 = v17 + 16LL * v20;
      v22 = *(_QWORD *)v21;
      if ( a2 == *(_QWORD *)v21 )
        goto LABEL_12;
      ++v18;
    }
    if ( !v19 )
      v19 = v21;
    ++*(_QWORD *)(a1 + 664);
    v66 = *(_DWORD *)(a1 + 680);
    v125 = (_BYTE *)v19;
    v67 = v66 + 1;
    if ( 4 * (v66 + 1) < 3 * v16 )
    {
      v68 = a2;
      if ( v16 - *(_DWORD *)(a1 + 684) - v67 > v16 >> 3 )
      {
LABEL_91:
        *(_DWORD *)(a1 + 680) = v67;
        if ( *(_QWORD *)v19 != -4096 )
          --*(_DWORD *)(a1 + 684);
        *(_QWORD *)v19 = v68;
        *(_DWORD *)(v19 + 8) = 0;
        v104 = 0;
        goto LABEL_13;
      }
LABEL_96:
      sub_2E515B0(v11, v16);
      sub_2E50510(v11, v117.m128i_i64, &v125);
      v68 = v117.m128i_i64[0];
      v19 = (__int64)v125;
      v67 = *(_DWORD *)(a1 + 680) + 1;
      goto LABEL_91;
    }
LABEL_95:
    v16 *= 2;
    goto LABEL_96;
  }
LABEL_12:
  v104 = *(_DWORD *)(v21 + 8);
LABEL_13:
  v23 = v122;
  v125 = v127;
  v126 = 0x800000000LL;
  v107 = &v122[(unsigned int)v123];
  if ( v122 == v107 )
  {
    v47 = 0;
    goto LABEL_51;
  }
  v102 = a1 + 664;
  v24 = v122;
  v111 = 0;
  v25 = a3;
  do
  {
    v26 = *v24;
    v27 = *(_BYTE *)(a4 + 28) == 0;
    v117.m128i_i64[0] = *v24;
    if ( v27 )
    {
      if ( !sub_C8CA60(a4, v26) )
      {
LABEL_48:
        v46 = (unsigned __int64)v125;
        v47 = 0;
        goto LABEL_49;
      }
      v30 = v117.m128i_i64[0];
      v31 = *(_BYTE *)(v25 + 8) & 1;
      if ( v31 )
      {
LABEL_21:
        v32 = v25 + 16;
        v33 = 15;
        goto LABEL_22;
      }
    }
    else
    {
      v28 = *(__int64 **)(a4 + 8);
      v29 = &v28[*(unsigned int *)(a4 + 20)];
      if ( v28 == v29 )
        goto LABEL_48;
      while ( 1 )
      {
        v30 = *v28;
        if ( v26 == *v28 )
          break;
        if ( v29 == ++v28 )
          goto LABEL_48;
      }
      v31 = *(_BYTE *)(v25 + 8) & 1;
      if ( v31 )
        goto LABEL_21;
    }
    v60 = *(unsigned int *)(v25 + 24);
    v32 = *(_QWORD *)(v25 + 16);
    if ( !(_DWORD)v60 )
      goto LABEL_59;
    v33 = v60 - 1;
LABEL_22:
    v34 = v33 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v35 = (__int64 *)(v32 + 16LL * v34);
    v36 = *v35;
    if ( v30 == *v35 )
      goto LABEL_23;
    v62 = 1;
    while ( v36 != -4096 )
    {
      v65 = v62 + 1;
      v34 = v33 & (v62 + v34);
      v35 = (__int64 *)(v32 + 16LL * v34);
      v36 = *v35;
      if ( v30 == *v35 )
        goto LABEL_23;
      v62 = v65;
    }
    if ( v31 )
    {
      v61 = 256;
      goto LABEL_60;
    }
    v60 = *(unsigned int *)(v25 + 24);
LABEL_59:
    v61 = 16 * v60;
LABEL_60:
    v35 = (__int64 *)(v32 + v61);
LABEL_23:
    v37 = v35[1];
    v38 = *(_DWORD *)sub_2E51790(v102, v117.m128i_i64) < v104;
    v41 = (unsigned int)v126;
    v111 += v38;
    v42 = v117.m128i_i64[0];
    v43 = (unsigned int)v126 + 1LL;
    if ( v43 > HIDWORD(v126) )
    {
      v98 = v117.m128i_i64[0];
      sub_C8D5F0((__int64)&v125, v127, v43, 0x10u, v39, v40);
      v41 = (unsigned int)v126;
      v42 = v98;
    }
    v44 = (__int64 *)&v125[16 * v41];
    ++v24;
    *v44 = v42;
    v44[1] = v37;
    v45 = (unsigned int)(v126 + 1);
    LODWORD(v126) = v126 + 1;
  }
  while ( v107 != v24 );
  v46 = (unsigned __int64)v125;
  v47 = 0;
  if ( v45 )
  {
    v48 = (const __m128i *)*((_QWORD *)v125 + 1);
    if ( a5[3].m128i_i32[2] == 2 && a5[2].m128i_i32[1] == *(_DWORD *)(a2 + 24) )
    {
      v108 = v111;
      v49 = *((_QWORD *)v125 + 1);
      v112 = &v125[16 * v45];
      for ( i = v49; ; i = *(_QWORD *)(v46 + 8) )
      {
        v51 = sub_AF65F0(*(_QWORD *)(i + 40), *(_BYTE *)(i + 48), *(_QWORD *)(v49 + 40), *(_BYTE *)(v49 + 48));
        if ( !v51 )
          goto LABEL_48;
        v52 = *(_QWORD *)(v46 + 8);
        v53 = *(_DWORD *)(v52 + 56);
        if ( v53 == 3 )
          goto LABEL_48;
        if ( v53 == 2 )
        {
          if ( !*(_DWORD *)(v52 + 32) )
            goto LABEL_73;
          v54 = 0;
          if ( *(_DWORD *)(v49 + 56) != 2 )
            goto LABEL_44;
        }
        else
        {
          v54 = 0;
          if ( *(_DWORD *)(v49 + 56) != 2 )
            goto LABEL_44;
        }
        if ( *(_DWORD *)(v49 + 32) )
        {
          while ( 1 )
          {
LABEL_44:
            if ( *(_BYTE *)(v52 + 49) )
            {
              if ( v54 >= (unsigned int)sub_AF4EB0(*(_QWORD *)(v52 + 40)) )
                break;
            }
            else if ( v54 )
            {
              break;
            }
            if ( *(_DWORD *)(v52 + 32) )
              v55 = *(_DWORD *)(v52 + 4LL * v54);
            else
              v55 = unk_5051178;
            v56 = 0;
            if ( (v55 & 1) != 0 )
              v56 = unk_5051178 != v55;
            if ( *(_DWORD *)(v49 + 32) )
              v57 = *(_DWORD *)(v49 + 4LL * v54);
            else
              v57 = unk_5051178;
            v58 = 0;
            if ( (v57 & 1) != 0 )
              v58 = unk_5051178 != v57;
            if ( v56 != v58 )
              goto LABEL_48;
            ++v54;
          }
        }
LABEL_73:
        v46 += 16LL;
        if ( (_BYTE *)v46 == v112 )
        {
          v69 = v51;
          v46 = (unsigned __int64)v125;
          v70 = v49;
          v71 = v108;
          v72 = 16LL * (unsigned int)v126;
          v109 = &v125[v72];
          if ( &v125[v72] == v125 )
            goto LABEL_118;
          v115 = 0;
          v73 = (__int64)v125;
          v74 = v70;
          v105 = v71;
          v75 = (__int64)v125;
          v76 = v69;
          do
          {
            v77 = *(_DWORD **)(v75 + 8);
            if ( !(unsigned __int8)sub_37B9AB0((__int64)v77, v74) )
            {
              v78 = v77[8];
              if ( v78 )
              {
                v79 = *(unsigned int *)(v74 + 32);
                v80 = v79;
                if ( v79 == v78 )
                {
                  v81 = (_DWORD *)v74;
                  v82 = v77;
                  v83 = &v77[v80];
                  while ( *v82 == *v81 )
                  {
                    ++v82;
                    ++v81;
                    if ( v83 == v82 )
                      goto LABEL_108;
                  }
                }
              }
              if ( v77[14] == 2 && v77[9] == *(_DWORD *)(a2 + 24) )
              {
                v93 = v115;
                if ( v105 > (v75 - v73) >> 4 )
                  v93 = v76;
                v115 = v93;
              }
              else
              {
                v115 = v76;
              }
            }
LABEL_108:
            v75 += 16;
          }
          while ( (_BYTE *)v75 != v109 );
          v70 = v74;
          v69 = v76;
          v46 = v73;
          if ( !v115 )
          {
LABEL_118:
            v116 = v69;
            v94 = sub_37B9AB0((__int64)a5, v70);
            v47 = 0;
            if ( !v94 )
            {
              v47 = v116;
              *a5 = _mm_loadu_si128((const __m128i *)v70);
              a5[1] = _mm_loadu_si128((const __m128i *)(v70 + 16));
              a5[2] = _mm_loadu_si128((const __m128i *)(v70 + 32));
              a5[3].m128i_i64[0] = *(_QWORD *)(v70 + 48);
              a5[3].m128i_i32[2] = *(_DWORD *)(v70 + 56);
            }
          }
          else
          {
            v84 = *(_DWORD *)(a2 + 24);
            v85 = unk_5051178;
            v86 = &v117;
            do
            {
              v86->m128i_i32[0] = v85;
              v86 = (__m128i *)((char *)v86 + 4);
            }
            while ( v86 != (__m128i *)v119 );
            v119[1] = v84;
            v119[0] = 0;
            v87 = _mm_loadu_si128((const __m128i *)(v70 + 40));
            v121 = 2;
            v120 = v87;
            v88 = sub_37B9AB0((__int64)a5, (__int64)&v117);
            LOBYTE(v88) = v88 ^ 1;
            v47 = v88;
            if ( (_BYTE)v88 )
            {
              v89 = _mm_load_si128(&v117);
              v90 = _mm_load_si128(&v118);
              a5[3].m128i_i32[2] = 2;
              si128 = _mm_load_si128((const __m128i *)v119);
              v92 = v120.m128i_i64[1];
              *a5 = v89;
              a5[3].m128i_i64[0] = v92;
              a5[1] = v90;
              a5[2] = si128;
            }
          }
          goto LABEL_49;
        }
      }
    }
    v64 = sub_37B9AB0((__int64)a5, *((_QWORD *)v125 + 1));
    v47 = 0;
    if ( !v64 )
    {
      v47 = 1;
      *a5 = _mm_loadu_si128(v48);
      a5[1] = _mm_loadu_si128(v48 + 1);
      a5[2] = _mm_loadu_si128(v48 + 2);
      a5[3].m128i_i64[0] = v48[3].m128i_i64[0];
      a5[3].m128i_i32[2] = v48[3].m128i_i32[2];
    }
  }
LABEL_49:
  v23 = v122;
  if ( (_BYTE *)v46 != v127 )
  {
    v113 = v47;
    _libc_free(v46);
    v23 = v122;
    v47 = v113;
  }
LABEL_51:
  if ( v23 != (__int64 *)v124 )
  {
    v114 = v47;
    _libc_free((unsigned __int64)v23);
    return v114;
  }
  return v47;
}
