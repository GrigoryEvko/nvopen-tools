// Function: sub_18DE770
// Address: 0x18de770
//
__int64 __fastcall sub_18DE770(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  char v6; // al
  unsigned int v7; // r8d
  char v9; // bl
  char v10; // al
  char v11; // dl
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdx
  _QWORD v16[6]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v17[10]; // [rsp+30h] [rbp-50h] BYREF

  v5 = *a1;
  v17[0] = a3;
  v16[0] = a2;
  v17[1] = -1;
  memset(&v17[2], 0, 24);
  v16[1] = -1;
  memset(&v16[2], 0, 24);
  v6 = sub_134CB50(v5, (__int64)v16, (__int64)v17);
  if ( !v6 )
    return 0;
  v7 = 1;
  if ( (unsigned __int8)(v6 - 2) <= 1u )
    return v7;
  v9 = sub_18CE260(a2);
  v10 = sub_18CE260(a3);
  if ( !v9 )
  {
    v11 = *(_BYTE *)(a2 + 16);
    if ( !v10 || v11 != 54 )
      goto LABEL_8;
    goto LABEL_17;
  }
  v14 = a2;
  if ( *(_BYTE *)(a3 + 16) == 54 )
    return sub_18DD490(v14);
  v11 = *(_BYTE *)(a2 + 16);
  if ( v10 )
  {
    v7 = 0;
    if ( v11 != 54 )
      return v7;
LABEL_17:
    v14 = a3;
    return sub_18DD490(v14);
  }
LABEL_8:
  if ( v11 == 77 )
  {
    v15 = a3;
  }
  else
  {
    v12 = *(_BYTE *)(a3 + 16);
    if ( v12 <= 0x17u )
    {
      v7 = 1;
      if ( v11 != 79 )
        return v7;
      goto LABEL_12;
    }
    if ( v12 != 77 )
    {
      if ( v11 != 79 )
      {
        v7 = 1;
        if ( v12 != 79 )
          return v7;
        v13 = a2;
        a2 = a3;
        return sub_18DE3D0((__int64)a1, a2, v13);
      }
LABEL_12:
      v13 = a3;
      return sub_18DE3D0((__int64)a1, a2, v13);
    }
    v15 = a2;
    a2 = a3;
  }
  return sub_18DE480((__int64)a1, a2, v15);
}
