// Function: sub_688AA0
// Address: 0x688aa0
//
_BOOL8 sub_688AA0()
{
  _BOOL4 v0; // r13d
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v4; // rdx
  __int64 v5; // rcx
  _BYTE v6[48]; // [rsp+0h] [rbp-30h] BYREF

  v0 = 0;
  sub_7ADF70(v6, 0);
  sub_7AE360(v6);
  sub_7B8B50(v6, 0, v1, v2);
  if ( word_4F06418[0] == 27 )
  {
    sub_7AE360(v6);
    sub_7B8B50(v6, 0, v4, v5);
    v0 = sub_679C10(0x401u) != 0;
  }
  sub_7BC000(v6);
  return v0;
}
