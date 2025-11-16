// Function: sub_BB96F0
// Address: 0xbb96f0
//
__int64 __fastcall sub_BB96F0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  __int64 v5; // r8
  __int64 *v6; // rsi
  __int64 v7; // r8
  _QWORD *v8; // rdi
  __int64 v9; // r8
  __int64 *v10; // rsi
  __int64 v11; // r8
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD **)a1;
  v5 = *(unsigned int *)(a1 + 8);
  v13[0] = a2;
  v6 = &v4[v5];
  if ( v6 == sub_BB8800(v4, (__int64)v6, v13) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, a1 + 16, v7 + 1, 8);
      v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
    *v6 = a2;
    ++*(_DWORD *)(a1 + 8);
  }
  v8 = *(_QWORD **)(a1 + 80);
  v9 = *(unsigned int *)(a1 + 88);
  v13[0] = a2;
  v10 = &v8[v9];
  if ( v10 != sub_BB8800(v8, (__int64)v10, v13) )
    return a1;
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    sub_C8D5F0(a1 + 80, a1 + 96, v11 + 1, 8);
    v10 = (__int64 *)(*(_QWORD *)(a1 + 80) + 8LL * *(unsigned int *)(a1 + 88));
  }
  *v10 = a2;
  ++*(_DWORD *)(a1 + 88);
  return a1;
}
