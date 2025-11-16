// Function: sub_2EF2A30
// Address: 0x2ef2a30
//
unsigned __int64 __fastcall sub_2EF2A30(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rax
  char v2; // dl
  char v3; // si
  unsigned __int64 v5; // rdx
  int v6; // ecx
  unsigned __int64 v7; // rax

  v1 = *a1;
  v2 = *(_BYTE *)a1;
  if ( (*a1 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
  {
    if ( (*(_BYTE *)a1 & 1) != 0 )
      return HIDWORD(v1);
    v5 = *a1 >> 3;
    v6 = (unsigned __int16)((unsigned int)v1 >> 8);
LABEL_8:
    v7 = v5 >> 29;
    return (unsigned int)(v6 * v7);
  }
  v3 = v2 & 2;
  if ( (v2 & 6) != 2 && (*(_BYTE *)a1 & 1) == 0 )
  {
    v7 = HIWORD(v1);
    v5 = *a1 >> 3;
    v6 = (unsigned __int16)((unsigned int)*a1 >> 8);
    if ( v3 )
      return (unsigned int)(v6 * v7);
    goto LABEL_8;
  }
  if ( !v3 )
    return HIDWORD(v1);
  return HIWORD(v1);
}
