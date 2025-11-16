// Function: sub_23A0E50
// Address: 0x23a0e50
//
__int64 __fastcall sub_23A0E50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 result; // rax
  _QWORD v8[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 368);
  v4 = v3 + 32LL * *(unsigned int *)(a1 + 376);
  if ( v4 != v3 )
  {
    v5 = a2;
    do
    {
      v8[0] = a3;
      if ( !*(_QWORD *)(v3 + 16) )
        sub_4263D6(a1, a2, a3);
      a1 = v3;
      a2 = v5;
      result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *))(v3 + 24))(v3, v5, v8);
      v3 += 32;
    }
    while ( v3 != v4 );
  }
  return result;
}
