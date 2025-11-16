// Function: sub_2C83AE0
// Address: 0x2c83ae0
//
char __fastcall sub_2C83AE0(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  char *v3; // rax
  char v4; // cl
  int v5; // r12d
  _BYTE *v7; // [rsp+0h] [rbp-20h]

  LOBYTE(v3) = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 != 61 )
  {
    switch ( (_BYTE)v3 )
    {
      case '>':
        v3 = *(char **)(*(_QWORD *)(a1 - 32) + 8LL);
        if ( v3[8] != 14 )
          return (char)v3;
LABEL_10:
        LODWORD(v3) = *((_DWORD *)v3 + 2);
        if ( (unsigned int)v3 > 0x1FF && (unsigned int)v3 >> 8 != 3 )
          return (char)v3;
LABEL_11:
        *a3 = 1;
        return (char)v3;
      case 'A':
        v3 = *(char **)(*(_QWORD *)(a1 - 96) + 8LL);
        if ( v3[8] != 14 )
          return (char)v3;
        LODWORD(v3) = *((_DWORD *)v3 + 2);
        if ( (unsigned int)v3 > 0x1FF && (unsigned int)v3 >> 8 != 3 )
          return (char)v3;
        goto LABEL_26;
      case 'B':
        v3 = *(char **)(*(_QWORD *)(a1 - 64) + 8LL);
        if ( v3[8] != 14 )
          return (char)v3;
        goto LABEL_10;
    }
    v7 = a3;
    if ( (_BYTE)v3 != 85 )
      return (char)v3;
    LOBYTE(v3) = sub_B49E00(a1);
    a3 = v7;
    if ( !(_BYTE)v3 )
    {
      v3 = *(char **)(a1 - 32);
      v4 = *v3;
      if ( *v3 )
      {
LABEL_34:
        if ( v4 == 25 && !v3[96] )
        {
          *a2 = 1;
          *a3 = 0;
          return (char)v3;
        }
        goto LABEL_26;
      }
      if ( *((_QWORD *)v3 + 3) != *(_QWORD *)(a1 + 80) || (v3[33] & 0x20) == 0 )
      {
LABEL_26:
        *a2 = 1;
        goto LABEL_11;
      }
      v5 = *((_DWORD *)v3 + 9);
      LOBYTE(v3) = sub_CEA1D0(v5);
      a3 = v7;
      if ( !(_BYTE)v3 )
      {
        LOBYTE(v3) = v5 == 292;
        if ( v5 != 292 && v5 != 376 && v5 != 7 )
        {
          LOBYTE(v3) = sub_CEA1A0(v5);
          a3 = v7;
          if ( !(_BYTE)v3 )
          {
            v3 = *(char **)(a1 - 32);
            v4 = *v3;
            goto LABEL_34;
          }
        }
      }
    }
    *a2 = 0;
    *a3 = 0;
    return (char)v3;
  }
  v3 = *(char **)(*(_QWORD *)(a1 - 32) + 8LL);
  if ( v3[8] == 14 )
  {
    LODWORD(v3) = *((_DWORD *)v3 + 2);
    if ( (unsigned int)v3 <= 0x1FF || (unsigned int)v3 >> 8 == 3 )
      *a2 = 1;
  }
  return (char)v3;
}
