// Function: sub_F7D1E0
// Address: 0xf7d1e0
//
__int64 __fastcall sub_F7D1E0(char *a1, unsigned __int8 *a2)
{
  char v2; // dl
  unsigned __int64 v3; // r13
  char v4; // dl
  int v5; // edx
  __int64 result; // rax
  int v7; // edx
  __int64 v8; // rax
  _BOOL4 v9; // edx

  v2 = *a1;
  *((_DWORD *)a1 + 1) = 0;
  v3 = *a2;
  v4 = v2 & 0xC0;
  *a1 = v4;
  if ( (unsigned __int8)v3 <= 0x36u )
  {
    v8 = 0x40540000000000LL;
    if ( _bittest64(&v8, v3) )
      *a1 = v4 | ((a2[1] & 2) != 0) | (a2[1] >> 1) & 2;
    if ( (unsigned int)(unsigned __int8)v3 - 48 > 1 )
      goto LABEL_12;
  }
  else if ( (unsigned __int8)(v3 - 55) > 1u )
  {
    goto LABEL_4;
  }
  *a1 = (2 * a2[1]) & 4 | *a1 & 0xFB;
LABEL_4:
  if ( (_BYTE)v3 == 58 )
  {
    v5 = (4 * a2[1]) & 8;
    result = v5 | *a1 & 0xF7u;
    *a1 = v5 | *a1 & 0xF7;
    goto LABEL_6;
  }
LABEL_12:
  result = (unsigned int)(v3 - 68);
  if ( (((_BYTE)v3 - 68) & 0xFB) == 0 )
  {
    v9 = 16 * sub_B44910((__int64)a2);
    result = v9 | *a1 & 0xEFu;
    *a1 = v9 | *a1 & 0xEF;
  }
  if ( (_BYTE)v3 == 67 )
  {
    result = *a1 & 0xFC | ((a2[1] & 2) != 0) | (a2[1] >> 1) & 2u;
    *a1 = *a1 & 0xFC | ((a2[1] & 2) != 0) | (a2[1] >> 1) & 2;
    return result;
  }
  if ( (_BYTE)v3 == 63 )
  {
    result = sub_B4DE20((__int64)a2);
    *((_DWORD *)a1 + 1) = result;
  }
LABEL_6:
  if ( *a2 == 82 )
  {
    v7 = (16 * a2[1]) & 0x20;
    result = v7 | *a1 & 0xDFu;
    *a1 = v7 | *a1 & 0xDF;
  }
  return result;
}
