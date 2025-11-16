// Function: sub_722680
// Address: 0x722680
//
__int64 __fastcall sub_722680(unsigned __int8 *a1, _QWORD *a2, int *a3, int a4)
{
  __int64 v5; // rax
  unsigned int v7; // r8d
  __int64 v9; // rax
  unsigned __int8 v10; // dl

  v5 = *a1;
  if ( a4 )
  {
    *a2 = v5;
    a4 = 0;
    v7 = 1;
    goto LABEL_5;
  }
  if ( (v5 & 0x80u) == 0LL )
  {
    *a2 = v5;
    v7 = 1;
    goto LABEL_5;
  }
  if ( (v5 & 0xE0) == 0xC0 )
  {
    if ( (a1[1] & 0xC0) == 0x80 )
    {
      *a2 = ((unsigned __int8)v5 << 6) & 0x7C0 | a1[1] & 0x3Fu;
      v7 = 2;
      goto LABEL_5;
    }
  }
  else if ( (v5 & 0xF0) == 0xE0 )
  {
    if ( (a1[1] & 0xC0) == 0x80 && (a1[2] & 0xC0) == 0x80 )
    {
      *a2 = (unsigned __int16)((unsigned __int8)v5 << 12) | (a1[1] << 6) & 0xFC0 | a1[2] & 0x3Fu;
      v7 = 3;
      goto LABEL_5;
    }
  }
  else if ( (v5 & 0xF8) == 0xF0 && (a1[1] & 0xC0) == 0x80 && (a1[2] & 0xC0) == 0x80 && (a1[3] & 0xC0) == 0x80 )
  {
    *a2 = (a1[2] << 6) & 0xFC0 | a1[3] & 0x3F | (a1[1] << 12) & 0x3F000 | ((unsigned __int8)v5 << 18) & 0x1C0000u;
    v7 = 4;
    goto LABEL_5;
  }
  *a2 = 0;
  if ( (a1[1] & 0xC0) == 0x80 )
  {
    v9 = 2;
    do
    {
      v10 = a1[v9];
      v7 = v9++;
    }
    while ( (v10 & 0xC0) == 0x80 );
    a4 = 1;
  }
  else
  {
    a4 = 1;
    v7 = 1;
  }
LABEL_5:
  if ( a3 )
    *a3 = a4;
  return v7;
}
