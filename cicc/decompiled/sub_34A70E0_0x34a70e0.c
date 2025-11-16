// Function: sub_34A70E0
// Address: 0x34a70e0
//
void __fastcall sub_34A70E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r10
  unsigned int v10; // r9d
  __int64 v11; // rbx
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char v16; // r13
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
  int v34; // r15d
  __int64 *v35; // rsi
  int v36; // eax
  int v37; // edx
  __int64 v38; // rcx
  unsigned int v39; // eax
  __int64 v40; // r13
  _QWORD *v41; // rax
  __int64 v42; // rsi
  __int64 **v43; // rdx
  __m128i v44; // [rsp+0h] [rbp-120h] BYREF
  __int64 v45; // [rsp+18h] [rbp-108h]
  __int64 **v46; // [rsp+20h] [rbp-100h]
  __int64 v47; // [rsp+28h] [rbp-F8h]
  const __m128i *v48; // [rsp+30h] [rbp-F0h]
  __int64 v49; // [rsp+38h] [rbp-E8h]
  __m128i v50; // [rsp+40h] [rbp-E0h] BYREF
  _QWORD v51[2]; // [rsp+58h] [rbp-C8h] BYREF
  char v52; // [rsp+68h] [rbp-B8h]
  __int64 v53[4]; // [rsp+80h] [rbp-A0h] BYREF
  char v54; // [rsp+A0h] [rbp-80h]
  __int64 *v55; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v56; // [rsp+B8h] [rbp-68h]
  _BYTE *v57; // [rsp+C8h] [rbp-58h] BYREF
  __int64 v58; // [rsp+D0h] [rbp-50h]
  _BYTE v59[72]; // [rsp+D8h] [rbp-48h] BYREF

  sub_B10CD0(a1 + 56);
  v5 = sub_2E891C0(a1);
  v49 = sub_2E89170(a1);
  if ( v5 && (sub_AF47B0((__int64)v51, *(unsigned __int64 **)(v5 + 16), *(unsigned __int64 **)(v5 + 24)), v52) )
  {
    v6 = v51[0];
    v7 = v51[1];
  }
  else
  {
    v6 = qword_4F81350[0];
    v7 = qword_4F81350[1];
  }
  v8 = *(unsigned int *)(a2 + 24);
  v50.m128i_i64[0] = v6;
  v50.m128i_i64[1] = v7;
  v53[0] = v49;
  if ( !(_DWORD)v8 )
  {
    ++*(_QWORD *)a2;
    v55 = 0;
    goto LABEL_52;
  }
  v9 = *(_QWORD *)(a2 + 8);
  LODWORD(v47) = ((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4);
  v10 = (v8 - 1) & v47;
  v11 = v9 + 136LL * v10;
  v12 = *(_QWORD *)v11;
  if ( v49 != *(_QWORD *)v11 )
  {
    v34 = 1;
    v35 = 0;
    while ( v12 != -4096 )
    {
      if ( !v35 && v12 == -8192 )
        v35 = (__int64 *)v11;
      v10 = (v8 - 1) & (v34 + v10);
      v11 = v9 + 136LL * v10;
      v12 = *(_QWORD *)v11;
      if ( v49 == *(_QWORD *)v11 )
        goto LABEL_6;
      ++v34;
    }
    v36 = *(_DWORD *)(a2 + 16);
    if ( !v35 )
      v35 = (__int64 *)v11;
    ++*(_QWORD *)a2;
    v37 = v36 + 1;
    v55 = v35;
    if ( 4 * (v36 + 1) < (unsigned int)(3 * v8) )
    {
      v38 = v49;
      v39 = v8 - *(_DWORD *)(a2 + 20) - v37;
      v46 = &v55;
      if ( v39 > (unsigned int)v8 >> 3 )
      {
LABEL_47:
        *(_DWORD *)(a2 + 16) = v37;
        if ( *v35 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v35 = v38;
        memset(v35 + 1, 0, 0x80u);
        v40 = (__int64)v46;
        v35[1] = (__int64)(v35 + 3);
        v35[2] = 0x400000000LL;
        v41 = v35 + 12;
        v42 = (__int64)(v35 + 1);
        *(_QWORD *)(v42 + 104) = v41;
        *(_QWORD *)(v42 + 112) = v41;
        sub_34A6AA0(v40, v42, &v50, 0, v8);
        v57 = v59;
        v55 = (__int64 *)v49;
        v56 = v50;
        v58 = 0x100000000LL;
        sub_34A6F40((__int64)v53, a3, v40, (__int64)&v57);
        if ( v57 != v59 )
          _libc_free((unsigned __int64)v57);
        return;
      }
      sub_34A5290(a2, v8);
      v43 = v46;
LABEL_53:
      sub_34A1EB0(a2, v53, v43);
      v38 = v53[0];
      v35 = v55;
      v37 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_47;
    }
LABEL_52:
    sub_34A5290(a2, 2 * v8);
    v46 = &v55;
    v43 = &v55;
    goto LABEL_53;
  }
LABEL_6:
  v56.m128i_i64[0] = v6;
  v56.m128i_i64[1] = v7;
  v55 = (__int64 *)v49;
  v46 = &v55;
  v57 = v59;
  v58 = 0x100000000LL;
  sub_34A6F40((__int64)v53, a3, (__int64)&v55, (__int64)&v57);
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  v16 = v54;
  if ( v54 )
  {
    v17 = *(_QWORD *)(v11 + 128) == 0;
    v18 = v53[2];
    v45 = v11 + 8;
    if ( v17 )
    {
      v19 = *(const __m128i **)(v11 + 8);
      v48 = &v19[*(unsigned int *)(v11 + 16)];
    }
    else
    {
      v19 = *(const __m128i **)(v11 + 112);
      v16 = 0;
      v48 = (const __m128i *)(v11 + 96);
    }
    v47 <<= 32;
    while ( 1 )
    {
      if ( v16 )
      {
        if ( v48 == v19 )
          goto LABEL_23;
        v33 = v19->m128i_u64[1];
        v13 = v50.m128i_i64[1] + v50.m128i_i64[0];
        if ( v50.m128i_i64[1] + v50.m128i_i64[0] <= v33 )
          goto LABEL_36;
        v21 = v19;
        if ( v19->m128i_i64[0] + v33 <= v50.m128i_i64[1] )
          goto LABEL_36;
      }
      else
      {
        while ( 1 )
        {
          if ( v48 == v19 )
          {
LABEL_23:
            sub_34A6AA0((__int64)v46, v45, &v50, v13, v14);
            return;
          }
          v20 = v19[2].m128i_u64[1];
          if ( v20 < v50.m128i_i64[1] + v50.m128i_i64[0] )
          {
            v21 = v19 + 2;
            if ( v19[2].m128i_i64[0] + v20 > v50.m128i_i64[1] )
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
        v44 = v23;
        sub_C8D5F0(v18 + 24, (const void *)(v18 + 40), v22 + 1, 0x10u, v14, v15);
        v22 = *(unsigned int *)(v18 + 32);
        v23 = _mm_load_si128(&v44);
      }
      *(__m128i *)(*(_QWORD *)(v18 + 24) + 16 * v22) = v23;
      ++*(_DWORD *)(v18 + 32);
      v24 = *(unsigned int *)(a3 + 24);
      v25 = v21->m128i_i64[0];
      v15 = v21->m128i_i64[1];
      v26 = *(_QWORD *)(a3 + 8);
      if ( (_DWORD)v24 )
      {
        v27 = 1;
        for ( i = (v24 - 1)
                & (((0xBF58476D1CE4E5B9LL * (v47 | (unsigned int)(unsigned __int16)v15 | ((_DWORD)v25 << 16))) >> 31)
                 ^ (484763065 * (v47 | (unsigned __int16)v15 | ((_DWORD)v25 << 16)))); ; i = (v24 - 1) & v30 )
        {
          v14 = i;
          v29 = v26 + 56LL * i;
          if ( v49 == *(_QWORD *)v29 && v25 == *(_QWORD *)(v29 + 8) && v15 == *(_QWORD *)(v29 + 16) )
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
      v32 = _mm_load_si128(&v50);
      if ( v31 + 1 > v13 )
      {
        v44 = v32;
        sub_C8D5F0(v29 + 24, (const void *)(v29 + 40), v31 + 1, 0x10u, v14, v15);
        v31 = *(unsigned int *)(v29 + 32);
        v32 = _mm_load_si128(&v44);
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
