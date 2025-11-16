// Function: sub_39A1860
// Address: 0x39a1860
//
__int64 __fastcall sub_39A1860(__int64 a1, __int64 a2, unsigned __int8 *a3, size_t a4)
{
  int v5; // r12d
  unsigned int v6; // r15d
  __int64 v7; // rbx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  _QWORD v13[2]; // [rsp+10h] [rbp-50h] BYREF
  __m128i v14; // [rsp+20h] [rbp-40h] BYREF

  v5 = a4;
  v13[0] = a3;
  v13[1] = a4;
  v14 = 0u;
  v6 = sub_16D19C0(a1, a3, a4);
  v7 = *(_QWORD *)a1 + 8LL * v6;
  if ( *(_QWORD *)v7 )
  {
    if ( *(_QWORD *)v7 != -8 )
      return *(_QWORD *)v7;
    --*(_DWORD *)(a1 + 16);
  }
  *(_QWORD *)v7 = sub_39A1620(a3, a4, *(__int64 **)(a1 + 24), &v14);
  ++*(_DWORD *)(a1 + 12);
  v7 = *(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v6);
  if ( *(_QWORD *)v7 == -8 || !*(_QWORD *)v7 )
  {
    v9 = (__int64 *)(v7 + 8);
    do
    {
      do
      {
        v10 = *v9;
        v7 = (__int64)v9++;
      }
      while ( v10 == -8 );
    }
    while ( !v10 );
  }
  v11 = *(_QWORD *)v7;
  *(_DWORD *)(v11 + 20) = *(_DWORD *)(a1 + 12) - 1;
  *(_DWORD *)(v11 + 16) = *(_DWORD *)(a1 + 48);
  if ( *(_BYTE *)(a1 + 52) )
  {
    v14.m128i_i16[0] = 261;
    v13[0] = a1 + 32;
    *(_QWORD *)(v11 + 8) = sub_396F530(a2, (__int64)v13);
  }
  else
  {
    *(_QWORD *)(v11 + 8) = 0;
  }
  *(_DWORD *)(a1 + 48) += v5 + 1;
  return *(_QWORD *)v7;
}
