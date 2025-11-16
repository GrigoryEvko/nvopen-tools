// Function: sub_1F3CE40
// Address: 0x1f3ce40
//
__int64 __fastcall sub_1F3CE40(__int64 a1, _QWORD **a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // r13
  const char *v5; // [rsp+0h] [rbp-40h] BYREF
  char v6; // [rsp+10h] [rbp-30h]
  char v7; // [rsp+11h] [rbp-2Fh]

  result = sub_1632000((__int64)a2, (__int64)"__stack_chk_guard", 17);
  if ( !result )
  {
    v3 = sub_16471D0(*a2, 0);
    v7 = 1;
    v5 = "__stack_chk_guard";
    v4 = v3;
    v6 = 3;
    result = (__int64)sub_1648A60(88, 1u);
    if ( result )
      return sub_15E51E0(result, (__int64)a2, v4, 0, 0, 0, (__int64)&v5, 0, 0, 0, 0);
  }
  return result;
}
