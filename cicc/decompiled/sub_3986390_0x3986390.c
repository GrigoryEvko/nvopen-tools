// Function: sub_3986390
// Address: 0x3986390
//
void __fastcall sub_3986390(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 v5; // rbx
  char v6; // al
  __int64 v7; // r14
  __int64 v8; // r9
  __m128i *v9; // rax
  unsigned int v10; // r11d
  __int64 v11; // rsi
  int v12; // edx
  int v13; // edx
  __int64 v14; // r8
  unsigned int v15; // r10d
  __int64 *v16; // rsi
  __int64 v17; // rdi
  unsigned int v18; // edi
  __m128i v19; // xmm0
  __int64 v20; // rcx
  __int64 v21; // rdx
  int v22; // edx
  unsigned int v23; // esi
  __int64 v24; // r10
  unsigned int v25; // edx
  int v26; // esi
  int v27; // [rsp+0h] [rbp-50h]
  int v28; // [rsp+8h] [rbp-48h]
  __int64 *v29; // [rsp+8h] [rbp-48h]
  int v30; // [rsp+8h] [rbp-48h]
  __int64 v32[7]; // [rsp+18h] [rbp-38h] BYREF

  v32[0] = a3;
  if ( src == a2 )
    return;
  v3 = src + 2;
  if ( a2 == src + 2 )
    return;
  do
  {
    while ( 1 )
    {
      v5 = *v3;
      v6 = sub_3985080((__int64)v32, *v3, src);
      v7 = v3[1];
      if ( !v6 )
        break;
      if ( src != v3 )
        memmove(src + 2, src, (char *)v3 - (char *)src);
      *src = v5;
      v3 += 2;
      src[1] = v7;
      if ( v3 == a2 )
        return;
    }
    v8 = v32[0];
    v9 = (__m128i *)v3;
    v10 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
    while ( 1 )
    {
      v20 = v9[-1].m128i_i64[0];
      if ( !v5 )
      {
        if ( !v20
          || (v21 = *(_QWORD *)(*(_QWORD *)(v8 + 8) + 256LL),
              v14 = *(_QWORD *)(v21 + 88),
              (v22 = *(_DWORD *)(v21 + 104)) == 0) )
        {
LABEL_26:
          v9->m128i_i64[0] = v5;
          v9->m128i_i64[1] = v7;
          goto LABEL_27;
        }
        v13 = v22 - 1;
LABEL_18:
        v23 = v13 & (((unsigned int)v20 >> 4) ^ ((unsigned int)v20 >> 9));
        v24 = *(_QWORD *)(v14 + 16LL * v23);
        if ( v20 == v24 )
          goto LABEL_26;
        v18 = 0;
LABEL_20:
        v28 = 1;
        while ( v24 != -8 )
        {
          v23 = v13 & (v28 + v23);
          v27 = v28 + 1;
          v29 = (__int64 *)(v14 + 16LL * v23);
          v24 = *v29;
          if ( v20 == *v29 )
            goto LABEL_23;
          v28 = v27;
        }
LABEL_12:
        if ( !v18 )
          goto LABEL_26;
        goto LABEL_13;
      }
      v11 = *(_QWORD *)(*(_QWORD *)(v8 + 8) + 256LL);
      v12 = *(_DWORD *)(v11 + 104);
      if ( !v12 )
        goto LABEL_26;
      v13 = v12 - 1;
      v14 = *(_QWORD *)(v11 + 88);
      v15 = v13 & v10;
      v16 = (__int64 *)(v14 + 16LL * (v13 & v10));
      v17 = *v16;
      if ( v5 != *v16 )
        break;
LABEL_11:
      v18 = *((_DWORD *)v16 + 2);
      if ( !v20 )
        goto LABEL_12;
      v23 = v13 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v29 = (__int64 *)(v14 + 16LL * v23);
      v24 = *v29;
      if ( v20 != *v29 )
        goto LABEL_20;
LABEL_23:
      v25 = *((_DWORD *)v29 + 2);
      if ( !v18 || v25 && v18 >= v25 )
        goto LABEL_26;
LABEL_13:
      v19 = _mm_loadu_si128(--v9);
      v9[1] = v19;
    }
    v26 = 1;
    while ( v17 != -8 )
    {
      v15 = v13 & (v26 + v15);
      v30 = v26 + 1;
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( v5 == *v16 )
        goto LABEL_11;
      v26 = v30;
    }
    if ( v20 )
      goto LABEL_18;
    v9->m128i_i64[0] = v5;
    v9->m128i_i64[1] = v7;
LABEL_27:
    v3 += 2;
  }
  while ( v3 != a2 );
}
