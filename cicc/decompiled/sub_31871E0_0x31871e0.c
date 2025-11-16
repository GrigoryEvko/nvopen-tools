// Function: sub_31871E0
// Address: 0x31871e0
//
__int64 __fastcall sub_31871E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  _QWORD v8[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 360);
  result = 5LL * *(unsigned int *)(a1 + 368);
  v5 = v3 + 40LL * *(unsigned int *)(a1 + 368);
  if ( v5 != v3 )
  {
    v6 = a2;
    do
    {
      v8[0] = a3;
      if ( !*(_QWORD *)(v3 + 24) )
        sub_4263D6(a1, a2, a3);
      a1 = v3 + 8;
      a2 = v6;
      result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *))(v3 + 32))(v3 + 8, v6, v8);
      v3 += 40;
    }
    while ( v3 != v5 );
  }
  return result;
}
