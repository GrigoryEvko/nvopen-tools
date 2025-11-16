// Function: sub_3245B60
// Address: 0x3245b60
//
__m128i *__fastcall sub_3245B60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // r9
  int v9; // r10d
  __int64 v10; // r8
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 *v15; // r12
  __int32 v16; // eax
  __m128i *result; // rax
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // eax
  int v24; // r10d
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdi
  unsigned int v28; // r15d
  __int64 v29; // rsi
  __m128i v30; // [rsp+0h] [rbp-40h] BYREF

  v6 = a1 + 336;
  v7 = *(_DWORD *)(a1 + 360);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 336);
    goto LABEL_23;
  }
  v8 = v7 - 1;
  v9 = 1;
  v10 = *(_QWORD *)(a1 + 344);
  v11 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = 0;
  v13 = (__int64 *)(v10 + 136LL * v11);
  v14 = *v13;
  if ( *v13 == a2 )
  {
LABEL_3:
    v15 = v13 + 1;
    goto LABEL_4;
  }
  while ( v14 != -4096 )
  {
    if ( !v12 && v14 == -8192 )
      v12 = v13;
    v11 = v8 & (v9 + v11);
    v13 = (__int64 *)(v10 + 136LL * v11);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_3;
    ++v9;
  }
  v18 = *(_DWORD *)(a1 + 352);
  if ( !v12 )
    v12 = v13;
  ++*(_QWORD *)(a1 + 336);
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v7 )
  {
LABEL_23:
    sub_3245760(v6, 2 * v7);
    v20 = *(_DWORD *)(a1 + 360);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 344);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = (__int64 *)(v22 + 136LL * v23);
      v10 = *v12;
      v19 = *(_DWORD *)(a1 + 352) + 1;
      if ( *v12 != a2 )
      {
        v24 = 1;
        v8 = 0;
        while ( v10 != -4096 )
        {
          if ( !v8 && v10 == -8192 )
            v8 = (__int64)v12;
          v23 = v21 & (v24 + v23);
          v12 = (__int64 *)(v22 + 136LL * v23);
          v10 = *v12;
          if ( *v12 == a2 )
            goto LABEL_19;
          ++v24;
        }
        if ( v8 )
          v12 = (__int64 *)v8;
      }
      goto LABEL_19;
    }
    goto LABEL_46;
  }
  v10 = v7 >> 3;
  if ( v7 - *(_DWORD *)(a1 + 356) - v19 <= (unsigned int)v10 )
  {
    sub_3245760(v6, v7);
    v25 = *(_DWORD *)(a1 + 360);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 344);
      v8 = 1;
      v10 = 0;
      v28 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = (__int64 *)(v27 + 136LL * v28);
      v29 = *v12;
      v19 = *(_DWORD *)(a1 + 352) + 1;
      if ( *v12 != a2 )
      {
        while ( v29 != -4096 )
        {
          if ( v29 == -8192 && !v10 )
            v10 = (__int64)v12;
          v28 = v26 & (v8 + v28);
          v12 = (__int64 *)(v27 + 136LL * v28);
          v29 = *v12;
          if ( *v12 == a2 )
            goto LABEL_19;
          v8 = (unsigned int)(v8 + 1);
        }
        if ( v10 )
          v12 = (__int64 *)v10;
      }
      goto LABEL_19;
    }
LABEL_46:
    ++*(_DWORD *)(a1 + 352);
    BUG();
  }
LABEL_19:
  *(_DWORD *)(a1 + 352) = v19;
  if ( *v12 != -4096 )
    --*(_DWORD *)(a1 + 356);
  *v12 = a2;
  memset(v12 + 1, 0, 0x80u);
  v15 = v12 + 1;
  v12[4] = (__int64)(v12 + 2);
  v12[5] = (__int64)(v12 + 2);
  v12[7] = (__int64)(v12 + 9);
  v12[8] = 0x800000000LL;
LABEL_4:
  v16 = *(unsigned __int16 *)(*(_QWORD *)(a3 + 8) + 20LL);
  if ( *(_WORD *)(*(_QWORD *)(a3 + 8) + 20LL) )
  {
    v30.m128i_i64[1] = a3;
    v30.m128i_i32[0] = v16;
    return sub_3245050(v15, &v30);
  }
  else
  {
    result = (__m128i *)*((unsigned int *)v15 + 14);
    if ( (unsigned __int64)result->m128i_u64 + 1 > *((unsigned int *)v15 + 15) )
    {
      sub_C8D5F0((__int64)(v15 + 6), v15 + 8, (unsigned __int64)result->m128i_u64 + 1, 8u, v10, v8);
      result = (__m128i *)*((unsigned int *)v15 + 14);
    }
    *(_QWORD *)(v15[6] + 8LL * (_QWORD)result) = a3;
    ++*((_DWORD *)v15 + 14);
  }
  return result;
}
