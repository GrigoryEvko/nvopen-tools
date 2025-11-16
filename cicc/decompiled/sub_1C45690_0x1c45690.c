// Function: sub_1C45690
// Address: 0x1c45690
//
char __fastcall sub_1C45690(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rdx
  char v9; // dl
  unsigned int v10; // r14d
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  LOBYTE(v3) = *(_BYTE *)(a1 + 16);
  if ( (_BYTE)v3 != 54 )
  {
    switch ( (_BYTE)v3 )
    {
      case '7':
        v3 = **(_QWORD **)(a1 - 24);
        if ( *(_BYTE *)(v3 + 8) != 15 )
          return v3;
LABEL_10:
        LODWORD(v3) = *(_DWORD *)(v3 + 8);
        if ( (unsigned int)v3 > 0x1FF && (unsigned int)v3 >> 8 != 3 )
          return v3;
LABEL_11:
        *a3 = 1;
        return v3;
      case ':':
        v3 = **(_QWORD **)(a1 - 72);
        if ( *(_BYTE *)(v3 + 8) != 15 )
          return v3;
        LODWORD(v3) = *(_DWORD *)(v3 + 8);
        if ( (unsigned int)v3 > 0x1FF && (unsigned int)v3 >> 8 != 3 )
          return v3;
        goto LABEL_15;
      case ';':
        v3 = **(_QWORD **)(a1 - 48);
        if ( *(_BYTE *)(v3 + 8) != 15 )
          return v3;
        goto LABEL_10;
    }
    if ( (_BYTE)v3 != 78 )
      return v3;
    LOBYTE(v3) = sub_1560260((_QWORD *)(a1 + 56), -1, 36);
    if ( (_BYTE)v3 )
    {
LABEL_31:
      *a2 = 0;
      *a3 = 0;
      return v3;
    }
    if ( *(char *)(a1 + 23) >= 0 )
      goto LABEL_29;
    v5 = sub_1648A40(a1);
    v7 = v5 + v6;
    v8 = 0;
    if ( *(char *)(a1 + 23) < 0 )
      v8 = sub_1648A40(a1);
    if ( !(unsigned int)((v7 - v8) >> 4) )
    {
LABEL_29:
      v3 = *(_QWORD *)(a1 - 24);
      v9 = *(_BYTE *)(v3 + 16);
      if ( v9 )
        goto LABEL_39;
      v12[0] = *(_QWORD *)(v3 + 112);
      LOBYTE(v3) = sub_1560260(v12, -1, 36);
      if ( (_BYTE)v3 )
        goto LABEL_31;
    }
    v3 = *(_QWORD *)(a1 - 24);
    v9 = *(_BYTE *)(v3 + 16);
    if ( !v9 )
    {
      if ( (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
      {
LABEL_15:
        *a2 = 1;
        goto LABEL_11;
      }
      v10 = *(_DWORD *)(v3 + 36);
      LOBYTE(v3) = sub_1C30240(v10);
      if ( (_BYTE)v3 )
        goto LABEL_31;
      LOBYTE(v3) = v10 == 149;
      if ( v10 == 149 || v10 == 215 )
        goto LABEL_31;
      if ( v10 == 3 )
        goto LABEL_31;
      LOBYTE(v3) = sub_1C301F0(v10);
      if ( (_BYTE)v3 )
        goto LABEL_31;
      v3 = *(_QWORD *)(a1 - 24);
      v9 = *(_BYTE *)(v3 + 16);
    }
LABEL_39:
    if ( v9 == 20 && !*(_BYTE *)(v3 + 96) )
    {
      *a2 = 1;
      *a3 = 0;
      return v3;
    }
    goto LABEL_15;
  }
  v3 = **(_QWORD **)(a1 - 24);
  if ( *(_BYTE *)(v3 + 8) == 15 )
  {
    LODWORD(v3) = *(_DWORD *)(v3 + 8);
    if ( (unsigned int)v3 <= 0x1FF || (unsigned int)v3 >> 8 == 3 )
      *a2 = 1;
  }
  return v3;
}
