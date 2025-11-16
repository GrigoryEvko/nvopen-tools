// Function: sub_E0E0E0
// Address: 0xe0e0e0
//
__int64 __fastcall sub_E0E0E0(__int64 a1)
{
  _BYTE *v1; // rax
  _BYTE *v2; // rdx
  unsigned int v3; // r8d
  _BYTE *v4; // rax
  char v6; // cl

  v1 = *(_BYTE **)a1;
  v2 = *(_BYTE **)(a1 + 8);
  v3 = 0;
  if ( *(_BYTE **)a1 == v2 )
    return v3;
  if ( *v1 == 114 )
  {
    *(_QWORD *)a1 = v1 + 1;
    if ( v2 == v1 + 1 )
      return 4;
    v6 = v1[1];
    v3 = 4;
    ++v1;
    if ( v6 != 86 )
      goto LABEL_4;
LABEL_8:
    v4 = v1 + 1;
    v3 |= 2u;
    *(_QWORD *)a1 = v4;
    if ( v2 == v4 || *v4 != 75 )
      return v3;
    goto LABEL_10;
  }
  if ( *v1 == 86 )
    goto LABEL_8;
LABEL_4:
  v4 = *(_BYTE **)a1;
  if ( **(_BYTE **)a1 != 75 )
    return v3;
LABEL_10:
  *(_QWORD *)a1 = v4 + 1;
  return v3 | 1;
}
