// Function: sub_E5B330
// Address: 0xe5b330
//
__int64 __fastcall sub_E5B330(_QWORD *a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 v5; // r15
  char v6; // al
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // rax
  _BYTE *v10; // rax
  const char *v11; // [rsp+0h] [rbp-60h] BYREF
  char v12; // [rsp+20h] [rbp-40h]
  char v13; // [rsp+21h] [rbp-3Fh]

  if ( !*(_BYTE *)(a1[39] + 21LL) )
    return sub_E97690();
  v4 = a1[1];
  v13 = 1;
  v11 = "debug_line_";
  v12 = 3;
  v5 = sub_E6C380(v4, &v11, 1);
  sub_E98820(a1, v5, 0);
  sub_EA12C0(v5, a1[38], a1[39]);
  sub_904010(a1[38], *(const char **)(a1[39] + 72LL));
  sub_E4D880((__int64)a1);
  v6 = *(_BYTE *)(v4 + 1906);
  if ( v6 )
  {
    if ( v6 != 1 )
      BUG();
    v7 = 12;
  }
  else
  {
    v7 = 4;
  }
  v8 = sub_E81A90(v7, v4, 0, 0);
  v9 = sub_E808D0(v5, 0, v4, 0);
  v10 = (_BYTE *)sub_E81A00(18, v9, v8, v4, 0);
  return sub_E5B0B0((__int64)a1, a2, v10);
}
