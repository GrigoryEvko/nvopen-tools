// Function: sub_23A12F0
// Address: 0x23a12f0
//
__int64 __fastcall sub_23A12F0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 result; // rax
  int v10; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 1168);
  v5 = v4 + 32LL * *(unsigned int *)(a1 + 1176);
  if ( v5 != v4 )
  {
    v6 = a2;
    do
    {
      v10 = a4;
      v11[0] = a3;
      if ( !*(_QWORD *)(v4 + 16) )
        sub_4263D6(a1, a2, a3);
      a1 = v4;
      a2 = v6;
      result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, int *))(v4 + 24))(v4, v6, v11, &v10);
      v4 += 32;
    }
    while ( v4 != v5 );
  }
  return result;
}
