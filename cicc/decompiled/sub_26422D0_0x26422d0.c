// Function: sub_26422D0
// Address: 0x26422d0
//
const __m128i **__fastcall sub_26422D0(__int64 a1, const __m128i *a2, __m128i *a3, __int64 a4)
{
  _QWORD *v7; // rdi
  __m128i *v8; // rax
  unsigned __int64 v9; // rdx
  __m128i *v10; // rsi
  _QWORD *v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r12
  const __m128i **v15; // rbx
  unsigned __int64 v16; // r9
  unsigned int v17; // r10d
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // r14
  __m128i *v22; // rcx
  const __m128i **result; // rax
  __m128i *v24; // r15
  __m128i *v25; // r14
  const __m128i *v26; // r9
  unsigned __int32 v27; // r10d
  __int64 v28; // rsi
  const __m128i *v30[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(_QWORD **)a1;
  v8 = (__m128i *)v7[2];
  if ( !v8 )
  {
    v10 = (__m128i *)(v7 + 1);
LABEL_10:
    v30[0] = a2;
    v10 = sub_263E8D0(v7, (__int64)v10, v30);
    goto LABEL_11;
  }
  v9 = a2->m128i_i64[0];
  v10 = (__m128i *)(v7 + 1);
  do
  {
    while ( v8[2].m128i_i64[0] >= v9
         && (v8[2].m128i_i64[0] != v9 || v8[2].m128i_i32[2] >= (unsigned __int32)a2->m128i_i32[2]) )
    {
      v10 = v8;
      v8 = (__m128i *)v8[1].m128i_i64[0];
      if ( !v8 )
        goto LABEL_8;
    }
    v8 = (__m128i *)v8[1].m128i_i64[1];
  }
  while ( v8 );
LABEL_8:
  if ( v7 + 1 == (_QWORD *)v10
    || v10[2].m128i_i64[0] > v9
    || v10[2].m128i_i64[0] == v9 && a2->m128i_i32[2] < (unsigned __int32)v10[2].m128i_i32[2] )
  {
    goto LABEL_10;
  }
LABEL_11:
  v10[3].m128i_i64[0] = a4;
  v11 = *(_QWORD **)(a1 + 8);
  v12 = v11[2];
  if ( !v12 )
  {
    v14 = (unsigned __int64)(v11 + 1);
LABEL_20:
    v30[0] = a2;
    v14 = sub_2642040(v11, v14, v30);
    goto LABEL_21;
  }
  v13 = a2->m128i_i64[0];
  v14 = (unsigned __int64)(v11 + 1);
  do
  {
    while ( *(_QWORD *)(v12 + 32) >= v13 && (*(_QWORD *)(v12 + 32) != v13 || *(_DWORD *)(v12 + 40) >= a2->m128i_i32[2]) )
    {
      v14 = v12;
      v12 = *(_QWORD *)(v12 + 16);
      if ( !v12 )
        goto LABEL_18;
    }
    v12 = *(_QWORD *)(v12 + 24);
  }
  while ( v12 );
LABEL_18:
  if ( v11 + 1 == (_QWORD *)v14
    || *(_QWORD *)(v14 + 32) > v13
    || *(_QWORD *)(v14 + 32) == v13 && a2->m128i_i32[2] < *(_DWORD *)(v14 + 40) )
  {
    goto LABEL_20;
  }
LABEL_21:
  v15 = (const __m128i **)(v14 + 56);
  if ( v14 + 56 != sub_263EB40(v14 + 48, (unsigned __int64 *)a3) )
  {
    v18 = *(_QWORD *)(v14 + 64);
    v19 = v14 + 56;
    if ( !v18 )
      goto LABEL_31;
    do
    {
      while ( v16 <= *(_QWORD *)(v18 + 32) && (v16 != *(_QWORD *)(v18 + 32) || v17 <= *(_DWORD *)(v18 + 40)) )
      {
        v19 = v18;
        v18 = *(_QWORD *)(v18 + 16);
        if ( !v18 )
          goto LABEL_29;
      }
      v18 = *(_QWORD *)(v18 + 24);
    }
    while ( v18 );
LABEL_29:
    if ( v15 == (const __m128i **)v19
      || v16 < *(_QWORD *)(v19 + 32)
      || v16 == *(_QWORD *)(v19 + 32) && v17 < *(_DWORD *)(v19 + 40) )
    {
LABEL_31:
      v30[0] = a3;
      v19 = sub_263EBB0((_QWORD *)(v14 + 48), v19, v30);
    }
    v16 = *(_QWORD *)(v19 + 48);
    v17 = *(_DWORD *)(v19 + 56);
  }
  *(_QWORD *)(a4 + 8) = v16;
  *(_DWORD *)(a4 + 16) = v17;
  v20 = **(_QWORD **)(a1 + 16);
  v21 = *(unsigned int *)(v20 + 32);
  v22 = *(__m128i **)(v20 + 24);
  result = v30;
  v24 = v22;
  v25 = &v22[v21];
  if ( v22 != v25 )
  {
    do
    {
      result = (const __m128i **)sub_263EB40(v14 + 48, (unsigned __int64 *)v24);
      if ( v15 != result )
      {
        result = *(const __m128i ***)(v14 + 64);
        v28 = v14 + 56;
        if ( !result )
          goto LABEL_46;
        do
        {
          while ( v26 <= result[4] && (v26 != result[4] || v27 <= *((_DWORD *)result + 10)) )
          {
            v28 = (__int64)result;
            result = (const __m128i **)result[2];
            if ( !result )
              goto LABEL_42;
          }
          result = (const __m128i **)result[3];
        }
        while ( result );
LABEL_42:
        if ( v15 == (const __m128i **)v28
          || (unsigned __int64)v26 < *(_QWORD *)(v28 + 32)
          || v26 == *(const __m128i **)(v28 + 32) && v27 < *(_DWORD *)(v28 + 40) )
        {
LABEL_46:
          v30[0] = v24;
          result = (const __m128i **)sub_263EBB0((_QWORD *)(v14 + 48), v28, v30);
          v28 = (__int64)result;
        }
        v26 = *(const __m128i **)(v28 + 48);
        v27 = *(_DWORD *)(v28 + 56);
      }
      v24->m128i_i64[0] = (__int64)v26;
      ++v24;
      v24[-1].m128i_i32[2] = v27;
    }
    while ( v25 != v24 );
  }
  return result;
}
