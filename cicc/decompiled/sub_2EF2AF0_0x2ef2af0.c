// Function: sub_2EF2AF0
// Address: 0x2ef2af0
//
unsigned __int64 __fastcall sub_2EF2AF0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rax
  char v2; // dl
  char v3; // di
  unsigned __int64 v4; // rax
  unsigned __int64 v6; // rcx
  int v7; // esi
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned int v10; // esi

  v1 = *a1;
  v2 = *(_BYTE *)a1;
  if ( (*a1 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    v3 = v2 & 2;
    if ( (v2 & 6) == 2 || (v2 & 1) != 0 )
    {
      if ( v3 )
      {
        v4 = HIWORD(v1);
        return (v4 + 7) >> 3;
      }
      goto LABEL_5;
    }
    v9 = v1;
    v10 = v1;
    v8 = HIWORD(v1);
    v6 = v9 >> 3;
    v7 = (unsigned __int16)(v10 >> 8);
    if ( v3 )
    {
LABEL_10:
      v4 = (unsigned int)(v7 * v8);
      return (v4 + 7) >> 3;
    }
LABEL_9:
    v8 = v6 >> 29;
    goto LABEL_10;
  }
  if ( (*(_BYTE *)a1 & 1) == 0 )
  {
    v6 = *a1 >> 3;
    v7 = (unsigned __int16)((unsigned int)v1 >> 8);
    goto LABEL_9;
  }
LABEL_5:
  v4 = HIDWORD(v1);
  return (v4 + 7) >> 3;
}
