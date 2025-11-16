// Function: sub_98CF40
// Address: 0x98cf40
//
__int64 __fastcall sub_98CF40(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r14
  unsigned int v6; // r14d
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 40);
  if ( v4 == *(_QWORD *)(a2 + 40) )
  {
    v6 = sub_B445A0();
    if ( !(_BYTE)v6 )
    {
      if ( a4 != 1 && a1 == a2 )
        return v6;
      LOWORD(v11) = 0;
      if ( !(unsigned __int8)sub_98CE60(15, a2, 0, v8, v9, v10, a2 + 24, v11, a1 + 24) )
        return v6;
      if ( !a4 )
        return (unsigned int)sub_984F60(a1, a2) ^ 1;
    }
  }
  else
  {
    if ( a3 )
      return sub_B19DB0(a3, a1, a2);
    if ( v4 != sub_AA54C0(*(_QWORD *)(a2 + 40)) )
      return sub_AA5B70(*(_QWORD *)(a1 + 40));
  }
  return 1;
}
