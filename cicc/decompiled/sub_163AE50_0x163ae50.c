// Function: sub_163AE50
// Address: 0x163ae50
//
__int64 __fastcall sub_163AE50(__int64 *a1, char *a2, __int64 a3)
{
  __int64 v4; // r14
  size_t v5; // rax
  __int64 v6; // rax
  __int64 v8[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = sub_1643360(a1);
  v5 = strlen(a2);
  v8[0] = sub_161FF10(a1, a2, v5);
  v6 = sub_15A0680(v4, a3, 0);
  v8[1] = (__int64)sub_1624210(v6);
  return sub_1627350(a1, v8, (__int64 *)2, 0, 1);
}
