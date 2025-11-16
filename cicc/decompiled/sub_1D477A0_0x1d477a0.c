// Function: sub_1D477A0
// Address: 0x1d477a0
//
__int64 *__fastcall sub_1D477A0(__int64 **a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 *v8; // r12
  __int64 *result; // rax
  __int64 v10; // rdi
  __int64 *v11; // rsi
  unsigned int v12; // r8d
  char v13; // dl
  __int64 v14; // rdx
  const __m128i *v15; // r13
  const __m128i *v16; // rbx
  __int64 v17; // rax
  bool v18; // zf
  __int64 *v19; // rcx
  __int64 v20; // r12
  __int64 v21; // rax
  _OWORD v22[3]; // [rsp+0h] [rbp-30h] BYREF

  v6 = a2[1];
  v7 = *a2;
  v8 = *a1;
  result = (__int64 *)(*(_QWORD *)(*a2 + 40) + 16LL * (unsigned int)v6);
  if ( *(_BYTE *)result != 1 || *(_WORD *)(v7 + 24) == 1 )
    return result;
  v10 = *v8;
  result = *(__int64 **)(*v8 + 8);
  if ( *(__int64 **)(*v8 + 16) != result )
    goto LABEL_4;
  v11 = &result[*(unsigned int *)(v10 + 28)];
  v12 = *(_DWORD *)(v10 + 28);
  if ( result == v11 )
  {
LABEL_23:
    if ( v12 < *(_DWORD *)(v10 + 24) )
    {
      *(_DWORD *)(v10 + 28) = ++v12;
      *v11 = v7;
      ++*(_QWORD *)v10;
LABEL_5:
      if ( *(_WORD *)(v7 + 24) == 2 )
      {
        result = *(__int64 **)(v7 + 32);
        v14 = 5LL * *(unsigned int *)(v7 + 56);
        v15 = (const __m128i *)&result[5 * *(unsigned int *)(v7 + 56)];
        if ( result != (__int64 *)v15 )
        {
          v16 = *(const __m128i **)(v7 + 32);
          do
          {
            v17 = v8[1];
            v18 = *(_QWORD *)(v17 + 16) == 0;
            v22[0] = _mm_loadu_si128(v16);
            if ( v18 )
              sub_4263D6(v10, v11, v14);
            v16 = (const __m128i *)((char *)v16 + 40);
            v11 = (__int64 *)v22;
            v10 = v17;
            result = (__int64 *)(*(__int64 (__fastcall **)(__int64, _OWORD *))(v17 + 24))(v17, v22);
          }
          while ( v15 != v16 );
        }
      }
      else
      {
        v20 = v8[2];
        v21 = *(unsigned int *)(v20 + 8);
        if ( (unsigned int)v21 >= *(_DWORD *)(v20 + 12) )
        {
          sub_16CD150(v20, (const void *)(v20 + 16), 0, 16, v12, a6);
          v21 = *(unsigned int *)(v20 + 8);
        }
        result = (__int64 *)(*(_QWORD *)v20 + 16 * v21);
        *result = v7;
        result[1] = v6;
        ++*(_DWORD *)(v20 + 8);
      }
      return result;
    }
LABEL_4:
    v11 = (__int64 *)v7;
    result = sub_16CCBA0(v10, v7);
    if ( !v13 )
      return result;
    goto LABEL_5;
  }
  v19 = 0;
  while ( v7 != *result )
  {
    if ( *result == -2 )
      v19 = result;
    if ( v11 == ++result )
    {
      if ( !v19 )
        goto LABEL_23;
      *v19 = v7;
      --*(_DWORD *)(v10 + 32);
      ++*(_QWORD *)v10;
      goto LABEL_5;
    }
  }
  return result;
}
