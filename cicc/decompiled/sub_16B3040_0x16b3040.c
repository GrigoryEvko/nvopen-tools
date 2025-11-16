// Function: sub_16B3040
// Address: 0x16b3040
//
__int64 __fastcall sub_16B3040(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _BYTE *a7)
{
  __int64 v7; // rax
  __int64 v8; // r9
  _QWORD v10[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v11[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v12; // [rsp+20h] [rbp-40h]
  _QWORD v13[2]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v14; // [rsp+40h] [rbp-20h]

  v10[1] = a6;
  v10[0] = a5;
  if ( !a6 )
    goto LABEL_7;
  if ( a6 == 4 )
  {
    if ( *(_DWORD *)v10[0] != 1702195828 && *(_DWORD *)v10[0] != 1163219540 && *(_DWORD *)v10[0] != 1702195796 )
      goto LABEL_5;
    goto LABEL_7;
  }
  if ( a6 != 1 )
  {
    if ( a6 == 5
      && (*(_DWORD *)v10[0] == 1936482662 && *(_BYTE *)(v10[0] + 4LL) == 101
       || *(_DWORD *)v10[0] == 1397506374 && *(_BYTE *)(v10[0] + 4LL) == 69
       || *(_DWORD *)v10[0] == 1936482630 && *(_BYTE *)(v10[0] + 4LL) == 101) )
    {
      goto LABEL_13;
    }
LABEL_5:
    v7 = sub_16E8CB0(a1, a2, v10[0]);
    v14 = 770;
    v12 = 1283;
    v11[0] = "'";
    v11[1] = v10;
    v13[0] = v11;
    v13[1] = "' is invalid value for boolean argument! Try 0 or 1";
    return sub_16B1F90(a2, (__int64)v13, 0, 0, v7, v8);
  }
  if ( *(_BYTE *)v10[0] == 49 )
  {
LABEL_7:
    *a7 = 1;
    return 0;
  }
  if ( *(_BYTE *)v10[0] != 48 )
    goto LABEL_5;
LABEL_13:
  *a7 = 0;
  return 0;
}
