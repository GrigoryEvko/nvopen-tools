// Function: sub_EF0590
// Address: 0xef0590
//
__int64 __fastcall sub_EF0590(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  _BYTE *v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // [rsp+0h] [rbp-20h] BYREF
  __int64 v10; // [rsp+8h] [rbp-18h] BYREF

  result = sub_EE6C50(a1);
  v9 = result;
  if ( result )
  {
    v5 = (_BYTE *)*a1;
    if ( *a1 != a1[1] && *v5 == 73 )
    {
      result = sub_EEFA10((__int64)a1, 0, (__int64)v5, v2, v3, v4);
      v10 = result;
      if ( result )
        return sub_EE7CC0((__int64)(a1 + 101), &v9, (unsigned __int64 *)&v10, v6, v7, v8);
    }
  }
  return result;
}
