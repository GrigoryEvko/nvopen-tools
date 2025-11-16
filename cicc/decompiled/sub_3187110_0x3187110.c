// Function: sub_3187110
// Address: 0x3187110
//
__int64 __fastcall sub_3187110(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax
  __int64 v5; // r13
  _QWORD *v6; // r12
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a1 + 264);
  result = 5LL * *(unsigned int *)(a1 + 272);
  v5 = v3 + 40LL * *(unsigned int *)(a1 + 272);
  if ( v5 != v3 )
  {
    v6 = a2;
    do
    {
      v7[0] = v6;
      if ( !*(_QWORD *)(v3 + 24) )
        sub_4263D6(a1, a2, a3);
      a1 = v3 + 8;
      a2 = v7;
      result = (*(__int64 (__fastcall **)(__int64, _QWORD *))(v3 + 32))(v3 + 8, v7);
      v3 += 40;
    }
    while ( v3 != v5 );
  }
  return result;
}
