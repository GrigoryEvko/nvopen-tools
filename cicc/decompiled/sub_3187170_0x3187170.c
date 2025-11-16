// Function: sub_3187170
// Address: 0x3187170
//
__int64 __fastcall sub_3187170(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax
  __int64 v5; // r14
  _QWORD *v6; // r12
  _QWORD v8[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 312);
  result = 5LL * *(unsigned int *)(a1 + 320);
  v5 = v3 + 40LL * *(unsigned int *)(a1 + 320);
  if ( v5 != v3 )
  {
    v6 = a2;
    do
    {
      v8[0] = v6;
      if ( !*(_QWORD *)(v3 + 24) )
        sub_4263D6(a1, a2, a3);
      a1 = v3 + 8;
      a2 = v8;
      result = (*(__int64 (__fastcall **)(__int64, _QWORD *, __int64))(v3 + 32))(v3 + 8, v8, a3);
      v3 += 40;
    }
    while ( v3 != v5 );
  }
  return result;
}
