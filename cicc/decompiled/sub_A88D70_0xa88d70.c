// Function: sub_A88D70
// Address: 0xa88d70
//
__int64 **__fastcall sub_A88D70(__int64 **a1)
{
  _DWORD *v1; // rax
  __int64 v2; // rdx
  char *v3; // rcx
  size_t v4; // r8
  __int64 **result; // rax
  int *v6; // rax
  __int64 v7; // rdx
  int v8; // ebx
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v9[0] = sub_A74E70((__int64)a1, "no-frame-pointer-elim", 0x15u);
  if ( !v9[0] )
  {
    if ( !sub_A75060((__int64)a1, "no-frame-pointer-elim-non-leaf", 0x1Eu) )
      goto LABEL_12;
LABEL_10:
    sub_A77740((__int64)a1, "no-frame-pointer-elim-non-leaf", 30);
    v3 = "non-leaf";
    v4 = 8;
    goto LABEL_11;
  }
  v1 = (_DWORD *)sub_A72240(v9);
  if ( v2 == 4 && *v1 == 1702195828 )
  {
    sub_A77740((__int64)a1, "no-frame-pointer-elim", 21);
    if ( sub_A75060((__int64)a1, "no-frame-pointer-elim-non-leaf", 0x1Eu) )
      sub_A77740((__int64)a1, "no-frame-pointer-elim-non-leaf", 30);
    v3 = "all";
    v4 = 3;
    goto LABEL_11;
  }
  sub_A77740((__int64)a1, "no-frame-pointer-elim", 21);
  if ( sub_A75060((__int64)a1, "no-frame-pointer-elim-non-leaf", 0x1Eu) )
    goto LABEL_10;
  v3 = "none";
  v4 = 4;
LABEL_11:
  sub_A78980(a1, "frame-pointer", 0xDu, v3, v4);
LABEL_12:
  result = (__int64 **)sub_A74E70((__int64)a1, "null-pointer-is-valid", 0x15u);
  v9[0] = (__int64)result;
  if ( result )
  {
    v6 = (int *)sub_A72240(v9);
    if ( v7 == 4 )
    {
      v8 = *v6;
      result = (__int64 **)sub_A77740((__int64)a1, "null-pointer-is-valid", 21);
      if ( v8 == 1702195828 )
        return sub_A77B20(a1, 44);
    }
    else
    {
      return (__int64 **)sub_A77740((__int64)a1, "null-pointer-is-valid", 21);
    }
  }
  return result;
}
