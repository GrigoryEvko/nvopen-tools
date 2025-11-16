// Function: sub_216F100
// Address: 0x216f100
//
__int64 __fastcall sub_216F100(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // r15
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10[8]; // [rsp+0h] [rbp-40h] BYREF

  if ( (*(_QWORD *)(a3 + 48) || *(__int16 *)(a3 + 18) < 0) && sub_1625790(a3, 4) )
    return 0;
  v5 = (__int64 *)sub_157E9C0(*(_QWORD *)(a3 + 40));
  v6 = sub_1643350(v5);
  v7 = sub_159C470(v6, a1, 0);
  v10[0] = (__int64)sub_1624210(v7);
  v8 = sub_159C470(v6, a2, 0);
  v10[1] = (__int64)sub_1624210(v8);
  v9 = sub_1627350(v5, v10, (__int64 *)2, 0, 1);
  sub_1625C10(a3, 4, v9);
  return 1;
}
