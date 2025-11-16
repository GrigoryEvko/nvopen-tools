// Function: sub_B32EC0
// Address: 0xb32ec0
//
__int64 __fastcall sub_B32EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  bool v7; // zf
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a4 )
    sub_B44240(a2, *(_QWORD *)(a4 + 16), a4, a5);
  sub_BD6B50(a2, a3);
  v7 = *(_QWORD *)(a1 + 24) == 0;
  v9[0] = a2;
  if ( v7 )
    sub_4263D6(a2, a3, v6);
  return (*(__int64 (__fastcall **)(__int64, _QWORD *))(a1 + 32))(a1 + 8, v9);
}
