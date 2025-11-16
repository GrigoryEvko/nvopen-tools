// Function: sub_BB9660
// Address: 0xbb9660
//
__int64 __fastcall sub_BB9660(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  __int64 v5; // r8
  __int64 *v6; // rsi
  __int64 v7; // r8
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(_QWORD **)a1;
  v5 = *(unsigned int *)(a1 + 8);
  v9[0] = a2;
  v6 = &v4[v5];
  if ( v6 != sub_BB8800(v4, (__int64)v6, v9) )
    return a1;
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, v7 + 1, 8);
    v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
  }
  *v6 = a2;
  ++*(_DWORD *)(a1 + 8);
  return a1;
}
