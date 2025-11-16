// Function: sub_15C88E0
// Address: 0x15c88e0
//
__int64 __fastcall sub_15C88E0(_BYTE *a1, unsigned __int64 a2)
{
  _WORD *v2; // rax
  unsigned int v3; // r12d
  _WORD *v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v6; // [rsp+8h] [rbp-28h]
  _WORD *v7; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v8; // [rsp+18h] [rbp-18h]

  v5 = a1;
  v6 = a2;
  if ( a2 )
  {
    if ( *a1 == 48 )
    {
      v8 = a2 - 1;
      v3 = 1;
      v7 = a1 + 1;
      if ( sub_16D24E0(&v7, "01234567", 8, 0) == -1 )
        return v3;
      a2 = v6;
    }
    if ( a2 > 1 )
    {
      v2 = v5;
      if ( *v5 != 28464 )
      {
LABEL_8:
        if ( *v2 == 30768 )
        {
          v8 = a2 - 2;
          v3 = 1;
          v7 = v2 + 1;
          if ( sub_16D24E0(&v7, "0123456789abcdefABCDEF", 22, 0) == -1 )
            return v3;
        }
        goto LABEL_10;
      }
      v8 = a2 - 2;
      v3 = 1;
      v7 = v5 + 1;
      if ( sub_16D24E0(&v7, "01234567", 8, 0) == -1 )
        return v3;
      a2 = v6;
      if ( v6 > 1 )
      {
        v2 = v5;
        goto LABEL_8;
      }
    }
  }
LABEL_10:
  v3 = 1;
  if ( sub_16D24E0(&v5, "0123456789", 10, 0) != -1
    && (v6 != 4 || *(_DWORD *)v5 != 1718511918 && *(_DWORD *)v5 != 1718503726 && *(_DWORD *)v5 != 1179535662) )
  {
    sub_16C9340(&v7, "^(\\.[0-9]+|[0-9]+(\\.[0-9]*)?)([eE][-+]?[0-9]+)?$", 48, 0);
    v3 = sub_16C9490(&v7, v5, v6, 0);
    sub_16C93F0(&v7);
  }
  return v3;
}
