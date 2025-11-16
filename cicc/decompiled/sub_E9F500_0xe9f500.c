// Function: sub_E9F500
// Address: 0xe9f500
//
__int64 __fastcall sub_E9F500(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 result; // rax
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // [rsp+8h] [rbp-38h]

  v4 = a4;
  for ( result = 0; result != 40; result += 8 )
    *(_QWORD *)(a1 + result) |= *(_QWORD *)(a2 + result);
  v7 = a3 + (a4 << 6);
  if ( a3 != v7 )
  {
    v8 = a3;
    result = 1;
    do
    {
      if ( (*(_QWORD *)(a2 + 8LL * (*(_DWORD *)(v8 + 16) >> 6)) & (1LL << *(_DWORD *)(v8 + 16))) != 0 )
      {
        v9 = v4;
        sub_E9F500(a1, v8 + 24, a3, v4);
        v4 = v9;
        result = 1;
      }
      v8 += 64;
    }
    while ( v7 != v8 );
  }
  return result;
}
