// Function: sub_2098400
// Address: 0x2098400
//
__int64 *__fastcall sub_2098400(__int64 a1, __int64 *a2, __int64 a3, __m128i a4, double a5, __m128i a6)
{
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // edx
  int v12; // r9d
  __int64 v13; // r15
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 *v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r14
  unsigned int v20; // edx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 *result; // rax
  __int64 v24; // rsi
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h] BYREF
  int v27; // [rsp+18h] [rbp-38h]

  v8 = *((_DWORD *)a2 + 134);
  v9 = *a2;
  v26 = 0;
  v27 = v8;
  if ( v9 )
  {
    if ( &v26 != (__int64 *)(v9 + 48) )
    {
      v10 = *(_QWORD *)(v9 + 48);
      v26 = v10;
      if ( v10 )
        sub_1623A60((__int64)&v26, v10, 2);
    }
  }
  v13 = sub_1D38BB0(a2[69], 2, (__int64)&v26, 6, 0, 1, a4, a5, a6, 0);
  v14 = v11;
  v15 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v15 >= *(_DWORD *)(a1 + 12) )
  {
    v25 = v11;
    sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, v11, v12);
    v15 = *(unsigned int *)(a1 + 8);
    v14 = v25;
  }
  v16 = (__int64 *)(*(_QWORD *)a1 + 16 * v15);
  v16[1] = v14;
  *v16 = v13;
  ++*(_DWORD *)(a1 + 8);
  v19 = sub_1D38BB0(a2[69], a3, (__int64)&v26, 6, 0, 1, a4, a5, a6, 0);
  v21 = v20;
  v22 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v22 >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, v17, v18);
    v22 = *(unsigned int *)(a1 + 8);
  }
  result = (__int64 *)(*(_QWORD *)a1 + 16 * v22);
  *result = v19;
  v24 = v26;
  result[1] = v21;
  ++*(_DWORD *)(a1 + 8);
  if ( v24 )
    return (__int64 *)sub_161E7C0((__int64)&v26, v24);
  return result;
}
