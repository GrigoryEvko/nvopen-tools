// Function: sub_1CD06C0
// Address: 0x1cd06c0
//
_BOOL8 __fastcall sub_1CD06C0(_BYTE *a1, char a2)
{
  char v2; // dl
  char v3; // cl
  _BOOL8 result; // rax
  const char *v5; // rsi
  const char *v6; // rsi
  __int64 v7; // r13
  __int64 v8; // rax
  int v9; // r13d
  __int64 v10[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = a1[16];
  v3 = v2 - 40;
  if ( (unsigned __int8)(v2 - 40) <= 0x25u )
  {
    result = ((0x20078C207FuLL >> v3) & 1) == 0;
    if ( ((0x20078C207FuLL >> v3) & 1) != 0 )
    {
      if ( a2 )
        return result;
      return 0;
    }
    if ( v2 != 54 )
    {
      if ( v2 == 72 )
      {
        v7 = *((_QWORD *)a1 - 3);
        if ( *(_BYTE *)(v7 + 16) == 17
          && !(unsigned __int8)sub_1C2F070(*(_QWORD *)(v7 + 24))
          && (unsigned __int8)sub_15E0420(v7, 6) )
        {
          return 0;
        }
      }
      goto LABEL_4;
    }
  }
  else
  {
    if ( v2 == 86 )
    {
      if ( byte_4FC0400 )
        return 1;
      if ( !a2 )
        return 0;
      v5 = "move instructions";
      goto LABEL_14;
    }
    if ( v2 != 78 )
    {
LABEL_4:
      if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 || *(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8 != 4 )
        return 1;
      if ( !a2 )
        return 0;
      v6 = "used in computing const address";
      goto LABEL_18;
    }
    v8 = *((_QWORD *)a1 - 3);
    if ( !*(_BYTE *)(v8 + 16) && (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
    {
      v9 = *(_DWORD *)(v8 + 36);
      if ( sub_1C30240(v9)
        || (unsigned int)(v9 - 4209) <= 1
        || (unsigned int)(v9 - 4286) <= 0x3E && ((0x5C07380000000007uLL >> ((unsigned __int8)v9 + 66)) & 1) != 0 )
      {
        goto LABEL_4;
      }
      if ( !a2 )
        return 0;
      v5 = "cost and aliases";
LABEL_14:
      sub_1CD0550(v10, v5);
      sub_2240A30(v10);
      return 0;
    }
  }
  if ( a2 )
  {
    v6 = "cost and aliases";
LABEL_18:
    sub_1CD0550(v10, v6);
    sub_2240A30(v10);
  }
  return 0;
}
