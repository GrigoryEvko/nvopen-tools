// Function: sub_1E62F30
// Address: 0x1e62f30
//
bool __fastcall sub_1E62F30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r15
  __int64 v6; // r14
  __int64 v7; // r12
  bool result; // al
  __int64 v10; // [rsp+10h] [rbp-40h]
  __int64 *v11; // [rsp+18h] [rbp-38h]

  v11 = *(__int64 **)(a2 + 72);
  if ( *(__int64 **)(a2 + 64) == v11 )
    return 1;
  v5 = *(__int64 **)(a2 + 64);
  while ( 1 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = *v5;
    sub_1E06620(v6);
    if ( sub_1E05550(*(_QWORD *)(v6 + 1312), a3, v7) )
    {
      v10 = *(_QWORD *)(a1 + 8);
      sub_1E06620(v10);
      result = sub_1E05550(*(_QWORD *)(v10 + 1312), a4, v7);
      if ( !result )
        break;
    }
    if ( v11 == ++v5 )
      return 1;
  }
  return result;
}
