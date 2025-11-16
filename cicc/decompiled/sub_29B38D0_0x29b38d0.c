// Function: sub_29B38D0
// Address: 0x29b38d0
//
__int64 __fastcall sub_29B38D0(__int64 a1, __int64 a2)
{
  __m128i v3; // xmm2
  __m128i v4; // xmm3
  __int64 v5; // rbx
  __int64 i; // rsi
  int v7; // edx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 result; // rax
  unsigned __int8 *v11; // r14
  unsigned int v12; // esi
  __int64 v13; // r8
  unsigned int v14; // edi
  __int64 *v15; // rdx
  __int64 v16; // rcx
  int v17; // r11d
  __int64 *v18; // rax
  int v19; // ecx
  int v20; // edx
  int v21; // r8d
  int v22; // r8d
  __int64 *v23; // rsi
  __int64 v24; // r10
  int v25; // r9d
  unsigned int v26; // ecx
  __int64 v27; // rdi
  __int64 v28; // rax
  unsigned int v29; // esi
  int v30; // edx
  __int64 v31; // r8
  unsigned int v32; // edi
  __int64 *v33; // r15
  __int64 *v34; // rax
  __int64 v35; // rcx
  unsigned int v36; // esi
  __int64 v37; // r11
  __int64 v38; // r10
  __int64 v39; // r8
  unsigned __int8 **v40; // rax
  int v41; // ecx
  unsigned __int8 **v42; // rdx
  int v43; // ecx
  int v44; // edx
  int v45; // r8d
  int v46; // r8d
  __int64 v47; // r10
  unsigned int v48; // ecx
  __int64 v49; // rdi
  int v50; // eax
  int v51; // eax
  int v52; // r9d
  int v53; // r9d
  __int64 v54; // r11
  unsigned int v55; // ecx
  unsigned __int8 *v56; // r8
  unsigned __int8 **v57; // rsi
  int v58; // r8d
  int v59; // r8d
  __int64 v60; // r9
  int v61; // ecx
  unsigned int v62; // r11d
  int v63; // edi
  int v64; // edi
  int v65; // edx
  __int64 *v66; // rsi
  __int64 v67; // r9
  unsigned int v68; // r8d
  __int64 v69; // rcx
  int v70; // edi
  int v71; // edi
  __int64 v72; // r9
  unsigned int v73; // r8d
  __int64 v74; // rcx
  int v75; // edx
  int v76; // r9d
  unsigned int v77; // [rsp+10h] [rbp-190h]
  __int64 v78; // [rsp+10h] [rbp-190h]
  __int64 v79; // [rsp+18h] [rbp-188h]
  unsigned int v80; // [rsp+24h] [rbp-17Ch]
  __int64 v82; // [rsp+38h] [rbp-168h] BYREF
  char v83[48]; // [rsp+40h] [rbp-160h] BYREF
  __m128i v84; // [rsp+70h] [rbp-130h]
  __m128i v85; // [rsp+80h] [rbp-120h]
  _BYTE v86[16]; // [rsp+90h] [rbp-110h] BYREF
  void (__fastcall *v87)(_BYTE *, _BYTE *, __int64); // [rsp+A0h] [rbp-100h]
  unsigned __int8 (__fastcall *v88)(_BYTE *, __int64); // [rsp+A8h] [rbp-F8h]
  __m128i v89; // [rsp+B0h] [rbp-F0h]
  __m128i v90; // [rsp+C0h] [rbp-E0h]
  _BYTE v91[16]; // [rsp+D0h] [rbp-D0h] BYREF
  void (__fastcall *v92)(_BYTE *, _BYTE *, __int64); // [rsp+E0h] [rbp-C0h]
  __int64 v93; // [rsp+E8h] [rbp-B8h]
  __m128i v94; // [rsp+F0h] [rbp-B0h] BYREF
  __m128i v95; // [rsp+100h] [rbp-A0h] BYREF
  _BYTE v96[16]; // [rsp+110h] [rbp-90h] BYREF
  void (__fastcall *v97)(_BYTE *, _BYTE *, __int64); // [rsp+120h] [rbp-80h]
  unsigned __int8 (__fastcall *v98)(_BYTE *, __int64); // [rsp+128h] [rbp-78h]
  __m128i v99; // [rsp+130h] [rbp-70h] BYREF
  __m128i v100; // [rsp+140h] [rbp-60h] BYREF
  _BYTE v101[16]; // [rsp+150h] [rbp-50h] BYREF
  void (__fastcall *v102)(_BYTE *, _BYTE *, __int64); // [rsp+160h] [rbp-40h]
  __int64 v103; // [rsp+168h] [rbp-38h]

  sub_AA72C0(&v94, a2, 1);
  v87 = 0;
  v84 = _mm_loadu_si128(&v94);
  v85 = _mm_loadu_si128(&v95);
  if ( v97 )
  {
    v97(v86, v96, 2);
    v88 = v98;
    v87 = v97;
  }
  v3 = _mm_loadu_si128(&v99);
  v4 = _mm_loadu_si128(&v100);
  v92 = 0;
  v89 = v3;
  v90 = v4;
  if ( v102 )
  {
    v102(v91, v101, 2);
    v93 = v103;
    v92 = v102;
  }
  v80 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v79 = a1 + 144;
LABEL_6:
  v5 = v84.m128i_i64[0];
  i = v84.m128i_i64[0];
  if ( v89.m128i_i64[0] == v84.m128i_i64[0] )
    goto LABEL_18;
  while ( 1 )
  {
    if ( !i )
      BUG();
    v7 = *(unsigned __int8 *)(i - 24);
    if ( (unsigned int)(v7 - 61) <= 1 )
      break;
    if ( (_BYTE)v7 == 85
      && (v28 = *(_QWORD *)(i - 56)) != 0
      && !*(_BYTE *)v28
      && *(_QWORD *)(v28 + 24) == *(_QWORD *)(i + 56)
      && (*(_BYTE *)(v28 + 33) & 0x20) != 0 )
    {
      v8 = i - 24;
      if ( !sub_B46A10(i - 24) )
      {
LABEL_48:
        v82 = a2;
        sub_29B09C0((__int64)v83, a1 + 176, &v82);
        goto LABEL_18;
      }
    }
    else
    {
      v8 = i - 24;
      if ( (unsigned __int8)sub_B46970((unsigned __int8 *)(i - 24)) )
        goto LABEL_48;
    }
LABEL_11:
    v5 = *(_QWORD *)(v5 + 8);
    v9 = 0;
    v84.m128i_i16[4] = 0;
    v84.m128i_i64[0] = v5;
    for ( i = v5; v85.m128i_i64[0] != i; v5 = i )
    {
      if ( i )
        i -= 24;
      if ( !v87 )
        sub_4263D6(v8, i, v9);
      v8 = (__int64)v86;
      if ( v88(v86, i) )
        goto LABEL_6;
      i = *(_QWORD *)(v84.m128i_i64[0] + 8);
      v84.m128i_i16[4] = 0;
      v84.m128i_i64[0] = i;
    }
    if ( v89.m128i_i64[0] == i )
      goto LABEL_18;
  }
  v8 = *(_QWORD *)(i - 56);
  if ( *(_BYTE *)v8 <= 0x15u )
    goto LABEL_11;
  v11 = sub_BD4070((unsigned __int8 *)v8, i);
  if ( *v11 == 60 )
  {
    v29 = *(_DWORD *)(a1 + 168);
    if ( v29 )
    {
      v30 = 1;
      v31 = *(_QWORD *)(a1 + 152);
      v32 = (v29 - 1) & v80;
      v33 = (__int64 *)(v31 + 40LL * v32);
      v34 = 0;
      v35 = *v33;
      if ( a2 == *v33 )
      {
LABEL_51:
        v36 = *((_DWORD *)v33 + 8);
        v37 = v33[2];
        v38 = (__int64)(v33 + 1);
        if ( v36 )
        {
          v5 = v84.m128i_i64[0];
          v77 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
          v39 = (v36 - 1) & v77;
          v40 = (unsigned __int8 **)(v37 + 8 * v39);
          v8 = (__int64)*v40;
          if ( v11 == *v40 )
            goto LABEL_11;
          v41 = 1;
          v42 = 0;
          while ( v8 != -4096 )
          {
            if ( !v42 && v8 == -8192 )
              v42 = v40;
            v39 = (v36 - 1) & ((_DWORD)v39 + v41);
            v40 = (unsigned __int8 **)(v37 + 8 * v39);
            v8 = (__int64)*v40;
            if ( v11 == *v40 )
              goto LABEL_62;
            ++v41;
          }
          v43 = *((_DWORD *)v33 + 6);
          if ( v42 )
            v40 = v42;
          ++v33[1];
          v44 = v43 + 1;
          if ( 4 * (v43 + 1) < 3 * v36 )
          {
            v8 = v36 >> 3;
            if ( v36 - *((_DWORD *)v33 + 7) - v44 > (unsigned int)v8 )
              goto LABEL_59;
            sub_CE2A30((__int64)(v33 + 1), v36);
            v58 = *((_DWORD *)v33 + 8);
            v38 = (__int64)(v33 + 1);
            if ( !v58 )
            {
LABEL_150:
              ++*(_DWORD *)(v38 + 16);
              BUG();
            }
            v59 = v58 - 1;
            v60 = v33[2];
            v57 = 0;
            v61 = 1;
            v44 = *((_DWORD *)v33 + 6) + 1;
            v62 = v59 & v77;
            v40 = (unsigned __int8 **)(v60 + 8LL * (v59 & v77));
            v8 = (__int64)*v40;
            if ( v11 == *v40 )
              goto LABEL_59;
            while ( v8 != -4096 )
            {
              if ( v8 == -8192 && !v57 )
                v57 = v40;
              v62 = v59 & (v61 + v62);
              v40 = (unsigned __int8 **)(v60 + 8LL * v62);
              v8 = (__int64)*v40;
              if ( v11 == *v40 )
                goto LABEL_59;
              ++v61;
            }
            goto LABEL_88;
          }
LABEL_84:
          v8 = v38;
          v78 = v38;
          sub_CE2A30(v38, 2 * v36);
          v38 = v78;
          v52 = *(_DWORD *)(v78 + 24);
          if ( !v52 )
            goto LABEL_150;
          v53 = v52 - 1;
          v54 = *(_QWORD *)(v78 + 8);
          v55 = v53 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v44 = *(_DWORD *)(v78 + 16) + 1;
          v40 = (unsigned __int8 **)(v54 + 8LL * v55);
          v56 = *v40;
          if ( v11 == *v40 )
            goto LABEL_59;
          v8 = 1;
          v57 = 0;
          while ( v56 != (unsigned __int8 *)-4096LL )
          {
            if ( v56 == (unsigned __int8 *)-8192LL && !v57 )
              v57 = v40;
            v55 = v53 & (v8 + v55);
            v40 = (unsigned __int8 **)(v54 + 8LL * v55);
            v56 = *v40;
            if ( v11 == *v40 )
              goto LABEL_59;
            v8 = (unsigned int)(v8 + 1);
          }
LABEL_88:
          if ( v57 )
            v40 = v57;
LABEL_59:
          *(_DWORD *)(v38 + 16) = v44;
          if ( *v40 != (unsigned __int8 *)-4096LL )
            --*(_DWORD *)(v38 + 20);
          *v40 = v11;
LABEL_62:
          v5 = v84.m128i_i64[0];
          goto LABEL_11;
        }
LABEL_83:
        ++*(_QWORD *)v38;
        v36 = 0;
        goto LABEL_84;
      }
      while ( v35 != -4096 )
      {
        if ( v35 == -8192 && !v34 )
          v34 = v33;
        v32 = (v29 - 1) & (v30 + v32);
        v33 = (__int64 *)(v31 + 40LL * v32);
        v35 = *v33;
        if ( a2 == *v33 )
          goto LABEL_51;
        ++v30;
      }
      if ( v34 )
        v33 = v34;
      v50 = *(_DWORD *)(a1 + 160);
      ++*(_QWORD *)(a1 + 144);
      v51 = v50 + 1;
      if ( 4 * v51 < 3 * v29 )
      {
        if ( v29 - *(_DWORD *)(a1 + 164) - v51 > v29 >> 3 )
          goto LABEL_80;
        sub_29AF080(v79, v29);
        v63 = *(_DWORD *)(a1 + 168);
        if ( !v63 )
          goto LABEL_149;
        v64 = v63 - 1;
        v65 = 1;
        v66 = 0;
        v67 = *(_QWORD *)(a1 + 152);
        v68 = v64 & v80;
        v33 = (__int64 *)(v67 + 40LL * (v64 & v80));
        v69 = *v33;
        v51 = *(_DWORD *)(a1 + 160) + 1;
        if ( a2 == *v33 )
          goto LABEL_80;
        while ( v69 != -4096 )
        {
          if ( v69 == -8192 && !v66 )
            v66 = v33;
          v68 = v64 & (v65 + v68);
          v33 = (__int64 *)(v67 + 40LL * v68);
          v69 = *v33;
          if ( a2 == *v33 )
            goto LABEL_80;
          ++v65;
        }
LABEL_102:
        if ( v66 )
          v33 = v66;
LABEL_80:
        *(_DWORD *)(a1 + 160) = v51;
        if ( *v33 != -4096 )
          --*(_DWORD *)(a1 + 164);
        v33[1] = 0;
        v38 = (__int64)(v33 + 1);
        v33[2] = 0;
        *v33 = a2;
        v33[3] = 0;
        *((_DWORD *)v33 + 8) = 0;
        goto LABEL_83;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 144);
    }
    sub_29AF080(v79, 2 * v29);
    v70 = *(_DWORD *)(a1 + 168);
    if ( !v70 )
    {
LABEL_149:
      ++*(_DWORD *)(a1 + 160);
      BUG();
    }
    v71 = v70 - 1;
    v72 = *(_QWORD *)(a1 + 152);
    v73 = v71 & v80;
    v33 = (__int64 *)(v72 + 40LL * (v71 & v80));
    v74 = *v33;
    v51 = *(_DWORD *)(a1 + 160) + 1;
    if ( a2 == *v33 )
      goto LABEL_80;
    v75 = 1;
    v66 = 0;
    while ( v74 != -4096 )
    {
      if ( !v66 && v74 == -8192 )
        v66 = v33;
      v73 = v71 & (v75 + v73);
      v33 = (__int64 *)(v72 + 40LL * v73);
      v74 = *v33;
      if ( a2 == *v33 )
        goto LABEL_80;
      ++v75;
    }
    goto LABEL_102;
  }
  v12 = *(_DWORD *)(a1 + 200);
  if ( v12 )
  {
    v13 = *(_QWORD *)(a1 + 184);
    v14 = (v12 - 1) & v80;
    v15 = (__int64 *)(v13 + 8LL * v14);
    v16 = *v15;
    if ( a2 == *v15 )
      goto LABEL_18;
    v17 = 1;
    v18 = 0;
    while ( v16 != -4096 )
    {
      if ( !v18 && v16 == -8192 )
        v18 = v15;
      v14 = (v12 - 1) & (v17 + v14);
      v15 = (__int64 *)(v13 + 8LL * v14);
      v16 = *v15;
      if ( a2 == *v15 )
        goto LABEL_18;
      ++v17;
    }
    v19 = *(_DWORD *)(a1 + 192);
    if ( !v18 )
      v18 = v15;
    ++*(_QWORD *)(a1 + 176);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(a1 + 196) - v20 > v12 >> 3 )
        goto LABEL_66;
      sub_CF28B0(a1 + 176, v12);
      v21 = *(_DWORD *)(a1 + 200);
      if ( v21 )
      {
        v22 = v21 - 1;
        v23 = 0;
        v24 = *(_QWORD *)(a1 + 184);
        v25 = 1;
        v26 = v22 & v80;
        v20 = *(_DWORD *)(a1 + 192) + 1;
        v18 = (__int64 *)(v24 + 8LL * (v22 & v80));
        v27 = *v18;
        if ( a2 != *v18 )
        {
          while ( v27 != -4096 )
          {
            if ( v27 == -8192 && !v23 )
              v23 = v18;
            v26 = v22 & (v25 + v26);
            v18 = (__int64 *)(v24 + 8LL * v26);
            v27 = *v18;
            if ( a2 == *v18 )
              goto LABEL_66;
            ++v25;
          }
          goto LABEL_40;
        }
        goto LABEL_66;
      }
LABEL_147:
      ++*(_DWORD *)(a1 + 192);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 176);
  }
  sub_CF28B0(a1 + 176, 2 * v12);
  v45 = *(_DWORD *)(a1 + 200);
  if ( !v45 )
    goto LABEL_147;
  v46 = v45 - 1;
  v47 = *(_QWORD *)(a1 + 184);
  v48 = v46 & v80;
  v20 = *(_DWORD *)(a1 + 192) + 1;
  v18 = (__int64 *)(v47 + 8LL * (v46 & v80));
  v49 = *v18;
  if ( a2 != *v18 )
  {
    v76 = 1;
    v23 = 0;
    while ( v49 != -4096 )
    {
      if ( !v23 && v49 == -8192 )
        v23 = v18;
      v48 = v46 & (v76 + v48);
      v18 = (__int64 *)(v47 + 8LL * v48);
      v49 = *v18;
      if ( a2 == *v18 )
        goto LABEL_66;
      ++v76;
    }
LABEL_40:
    if ( v23 )
      v18 = v23;
  }
LABEL_66:
  *(_DWORD *)(a1 + 192) = v20;
  if ( *v18 != -4096 )
    --*(_DWORD *)(a1 + 196);
  *v18 = a2;
LABEL_18:
  if ( v92 )
    v92(v91, v91, 3);
  if ( v87 )
    v87(v86, v86, 3);
  if ( v102 )
    v102(v101, v101, 3);
  result = (__int64)v97;
  if ( v97 )
    return ((__int64 (__fastcall *)(_BYTE *, _BYTE *, __int64))v97)(v96, v96, 3);
  return result;
}
