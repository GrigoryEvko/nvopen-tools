// Function: sub_37D4430
// Address: 0x37d4430
//
void __fastcall sub_37D4430(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // r8
  int v7; // r15d
  __int64 v8; // r10
  unsigned int v9; // r9d
  __int64 *v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char v16; // r14
  bool v17; // zf
  __int64 v18; // r15
  const __m128i *v19; // r12
  unsigned __int64 v20; // rax
  const __m128i *v21; // rbx
  __int64 v22; // rax
  __m128i v23; // xmm0
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rsi
  int v27; // r11d
  unsigned int i; // eax
  __int64 v29; // rbx
  unsigned int v30; // eax
  __int64 v31; // rax
  __m128i v32; // xmm0
  unsigned __int64 v33; // rax
  int v34; // eax
  int v35; // edx
  __int64 v36; // rcx
  unsigned int v37; // eax
  __int64 v38; // r14
  _QWORD *v39; // rax
  __int64 v40; // rsi
  __int64 **v41; // rdx
  __m128i v42; // [rsp+0h] [rbp-120h] BYREF
  __int64 v43; // [rsp+18h] [rbp-108h]
  __int64 **v44; // [rsp+20h] [rbp-100h]
  __int64 v45; // [rsp+28h] [rbp-F8h]
  const __m128i *v46; // [rsp+30h] [rbp-F0h]
  __int64 v47; // [rsp+38h] [rbp-E8h]
  __m128i v48; // [rsp+40h] [rbp-E0h] BYREF
  _QWORD v49[2]; // [rsp+58h] [rbp-C8h] BYREF
  char v50; // [rsp+68h] [rbp-B8h]
  __int64 v51[4]; // [rsp+80h] [rbp-A0h] BYREF
  char v52; // [rsp+A0h] [rbp-80h]
  __int64 *v53; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v54; // [rsp+B8h] [rbp-68h]
  _BYTE *v55; // [rsp+C8h] [rbp-58h] BYREF
  __int64 v56; // [rsp+D0h] [rbp-50h]
  _BYTE v57[72]; // [rsp+D8h] [rbp-48h] BYREF

  sub_B10CD0(a2 + 56);
  v3 = sub_2E891C0(a2);
  v47 = sub_2E89170(a2);
  if ( v3 && (sub_AF47B0((__int64)v49, *(unsigned __int64 **)(v3 + 16), *(unsigned __int64 **)(v3 + 24)), v50) )
  {
    v4 = v49[0];
    v5 = v49[1];
  }
  else
  {
    v4 = qword_4F81350[0];
    v5 = qword_4F81350[1];
  }
  v6 = *(unsigned int *)(a1 + 2128);
  v48.m128i_i64[0] = v4;
  v48.m128i_i64[1] = v5;
  v51[0] = v47;
  if ( !(_DWORD)v6 )
  {
    ++*(_QWORD *)(a1 + 2104);
    v53 = 0;
    goto LABEL_56;
  }
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 2112);
  LODWORD(v45) = ((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4);
  v9 = (v6 - 1) & v45;
  v10 = 0;
  v11 = v8 + 136LL * v9;
  v12 = *(_QWORD *)v11;
  if ( v47 != *(_QWORD *)v11 )
  {
    while ( v12 != -4096 )
    {
      if ( !v10 && v12 == -8192 )
        v10 = (__int64 *)v11;
      v9 = (v6 - 1) & (v7 + v9);
      v11 = v8 + 136LL * v9;
      v12 = *(_QWORD *)v11;
      if ( v47 == *(_QWORD *)v11 )
        goto LABEL_6;
      ++v7;
    }
    v34 = *(_DWORD *)(a1 + 2120);
    if ( !v10 )
      v10 = (__int64 *)v11;
    ++*(_QWORD *)(a1 + 2104);
    v35 = v34 + 1;
    v53 = v10;
    if ( 4 * (v34 + 1) < (unsigned int)(3 * v6) )
    {
      v36 = v47;
      v37 = v6 - *(_DWORD *)(a1 + 2124) - v35;
      v44 = &v53;
      if ( v37 > (unsigned int)v6 >> 3 )
      {
LABEL_51:
        *(_DWORD *)(a1 + 2120) = v35;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 2124);
        *v10 = v36;
        memset(v10 + 1, 0, 0x80u);
        v38 = (__int64)v44;
        v10[1] = (__int64)(v10 + 3);
        v10[2] = 0x400000000LL;
        v39 = v10 + 12;
        v40 = (__int64)(v10 + 1);
        *(_QWORD *)(v40 + 104) = v39;
        *(_QWORD *)(v40 + 112) = v39;
        sub_34A6AA0(v38, v40, &v48, 0, v6);
        v55 = v57;
        v53 = (__int64 *)v47;
        v54 = v48;
        v56 = 0x100000000LL;
        sub_34A6F40((__int64)v51, a1 + 2072, v38, (__int64)&v55);
        if ( v55 != v57 )
          _libc_free((unsigned __int64)v55);
        return;
      }
      sub_34A5290(a1 + 2104, v6);
      v41 = v44;
LABEL_57:
      sub_34A1EB0(a1 + 2104, v51, v41);
      v36 = v51[0];
      v10 = v53;
      v35 = *(_DWORD *)(a1 + 2120) + 1;
      goto LABEL_51;
    }
LABEL_56:
    sub_34A5290(a1 + 2104, 2 * v6);
    v44 = &v53;
    v41 = &v53;
    goto LABEL_57;
  }
LABEL_6:
  v54.m128i_i64[0] = v4;
  v54.m128i_i64[1] = v5;
  v53 = (__int64 *)v47;
  v44 = &v53;
  v55 = v57;
  v56 = 0x100000000LL;
  sub_34A6F40((__int64)v51, a1 + 2072, (__int64)&v53, (__int64)&v55);
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  v16 = v52;
  if ( v52 )
  {
    v17 = *(_QWORD *)(v11 + 128) == 0;
    v18 = v51[2];
    v43 = v11 + 8;
    if ( v17 )
    {
      v19 = *(const __m128i **)(v11 + 8);
      v46 = &v19[*(unsigned int *)(v11 + 16)];
    }
    else
    {
      v19 = *(const __m128i **)(v11 + 112);
      v16 = 0;
      v46 = (const __m128i *)(v11 + 96);
    }
    v45 <<= 32;
    while ( 1 )
    {
      if ( v16 )
      {
        if ( v46 == v19 )
          goto LABEL_23;
        v33 = v19->m128i_u64[1];
        v13 = v48.m128i_i64[1] + v48.m128i_i64[0];
        if ( v33 >= v48.m128i_i64[1] + v48.m128i_i64[0] )
          goto LABEL_36;
        v21 = v19;
        if ( v48.m128i_i64[1] >= v19->m128i_i64[0] + v33 )
          goto LABEL_36;
      }
      else
      {
        while ( 1 )
        {
          if ( v46 == v19 )
          {
LABEL_23:
            sub_34A6AA0((__int64)v44, v43, &v48, v13, v14);
            return;
          }
          v20 = v19[2].m128i_u64[1];
          if ( v48.m128i_i64[1] + v48.m128i_i64[0] > v20 )
          {
            v21 = v19 + 2;
            if ( v19[2].m128i_i64[0] + v20 > v48.m128i_i64[1] )
              break;
          }
LABEL_32:
          v19 = (const __m128i *)sub_220EF30((__int64)v19);
        }
      }
      v22 = *(unsigned int *)(v18 + 32);
      v23 = _mm_loadu_si128(v21);
      if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 36) )
      {
        v42 = v23;
        sub_C8D5F0(v18 + 24, (const void *)(v18 + 40), v22 + 1, 0x10u, v14, v15);
        v22 = *(unsigned int *)(v18 + 32);
        v23 = _mm_load_si128(&v42);
      }
      *(__m128i *)(*(_QWORD *)(v18 + 24) + 16 * v22) = v23;
      ++*(_DWORD *)(v18 + 32);
      v24 = *(unsigned int *)(a1 + 2096);
      v25 = v21->m128i_i64[0];
      v15 = v21->m128i_i64[1];
      v26 = *(_QWORD *)(a1 + 2080);
      if ( (_DWORD)v24 )
      {
        v27 = 1;
        for ( i = (v24 - 1)
                & (((0xBF58476D1CE4E5B9LL * (v45 | (unsigned int)(unsigned __int16)v15 | ((_DWORD)v25 << 16))) >> 31)
                 ^ (484763065 * (v45 | (unsigned __int16)v15 | ((_DWORD)v25 << 16)))); ; i = (v24 - 1) & v30 )
        {
          v14 = i;
          v29 = v26 + 56LL * i;
          if ( v47 == *(_QWORD *)v29 && v25 == *(_QWORD *)(v29 + 8) && v15 == *(_QWORD *)(v29 + 16) )
            break;
          if ( *(_QWORD *)v29 == -4096 && *(_QWORD *)(v29 + 8) == -1 && *(_QWORD *)(v29 + 16) == -1 )
            goto LABEL_28;
          v30 = v27 + i;
          ++v27;
        }
      }
      else
      {
LABEL_28:
        v29 = v26 + 56 * v24;
      }
      v31 = *(unsigned int *)(v29 + 32);
      v13 = *(unsigned int *)(v29 + 36);
      v32 = _mm_load_si128(&v48);
      if ( v31 + 1 > v13 )
      {
        v42 = v32;
        sub_C8D5F0(v29 + 24, (const void *)(v29 + 40), v31 + 1, 0x10u, v14, v15);
        v31 = *(unsigned int *)(v29 + 32);
        v32 = _mm_load_si128(&v42);
      }
      *(__m128i *)(*(_QWORD *)(v29 + 24) + 16 * v31) = v32;
      ++*(_DWORD *)(v29 + 32);
      if ( !v16 )
        goto LABEL_32;
LABEL_36:
      ++v19;
    }
  }
}
