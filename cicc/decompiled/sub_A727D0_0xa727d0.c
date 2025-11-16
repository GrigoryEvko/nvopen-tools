// Function: sub_A727D0
// Address: 0xa727d0
//
_DWORD *__fastcall sub_A727D0(__int64 a1, __int64 a2)
{
  _DWORD *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  v5[0] = sub_B2D7E0(a1, "unsafe-fp-math", 14);
  result = (_DWORD *)sub_A72240(v5);
  if ( v3 == 4 && *result == 1702195828 )
  {
    v5[0] = sub_B2D7E0(a2, "unsafe-fp-math", 14);
    result = (_DWORD *)sub_A72240(v5);
    if ( v4 != 4 || *result != 1702195828 )
      return (_DWORD *)sub_B2CD60(a1, "unsafe-fp-math", 14, "false", 5);
  }
  return result;
}
