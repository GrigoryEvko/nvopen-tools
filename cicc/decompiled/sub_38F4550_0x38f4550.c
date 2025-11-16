// Function: sub_38F4550
// Address: 0x38f4550
//
__int64 __fastcall sub_38F4550(__int64 a1)
{
  __int64 v1; // rdx
  unsigned int v2; // ecx
  __int64 v3; // rsi
  __int64 v5; // [rsp+0h] [rbp-70h] BYREF
  __int64 v6; // [rsp+8h] [rbp-68h]
  const char *v7; // [rsp+10h] [rbp-60h] BYREF
  char v8; // [rsp+20h] [rbp-50h]
  char v9; // [rsp+21h] [rbp-4Fh]
  _QWORD v10[2]; // [rsp+30h] [rbp-40h] BYREF
  char v11; // [rsp+40h] [rbp-30h]
  char v12; // [rsp+41h] [rbp-2Fh]

  v5 = 0;
  v6 = 0;
  if ( (unsigned __int8)sub_3909EB0(a1, 9) )
    goto LABEL_5;
  v9 = 1;
  v7 = "unexpected token";
  v8 = 3;
  v3 = 1;
  if ( !(unsigned __int8)sub_38F0EE0(a1, &v5, v1, v2) && v6 == 6 )
  {
    if ( *(_DWORD *)v5 != 1886218611 || (v3 = 0, *(_WORD *)(v5 + 4) != 25964) )
      v3 = 1;
  }
  if ( (unsigned __int8)sub_3909CB0(a1, v3, &v7)
    || (v12 = 1, v10[0] = "unexpected token", v11 = 3, (unsigned __int8)sub_3909E20(a1, 9, v10)) )
  {
    v12 = 1;
    v10[0] = " in '.cfi_startproc' directive";
    v11 = 3;
    return sub_39094A0(a1, v10);
  }
  else
  {
LABEL_5:
    sub_38E0040(*(_QWORD *)(a1 + 328), (const char **)(v6 != 0));
    return 0;
  }
}
