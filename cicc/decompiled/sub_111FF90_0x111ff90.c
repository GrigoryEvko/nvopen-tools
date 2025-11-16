// Function: sub_111FF90
// Address: 0x111ff90
//
_BOOL8 __fastcall sub_111FF90(_QWORD **a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // edx
  _BOOL8 result; // rax
  _BYTE *v6; // rdx
  __int64 v7; // rcx
  _BYTE *v8; // rdx
  __int64 v9; // rdx

  v2 = *a2;
  if ( (unsigned __int8)v2 <= 0x1Cu )
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v4 = *((unsigned __int16 *)a2 + 1);
    if ( (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD && (v4 & 0xFFF7) != 0x11 )
      return 0;
  }
  else
  {
    if ( (unsigned __int8)v2 > 0x36u )
      return 0;
    v3 = 0x40540000000000LL;
    v4 = (unsigned __int8)v2 - 29;
    if ( !_bittest64(&v3, v2) )
      return 0;
  }
  if ( v4 != 17 )
    return 0;
  result = (a2[1] & 2) != 0;
  if ( (a2[1] & 2) == 0 )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( *v6 != 68 )
    return 0;
  v7 = *((_QWORD *)v6 - 4);
  if ( !v7 )
    return 0;
  **a1 = v7;
  v8 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( *v8 != 68 )
    return 0;
  v9 = *((_QWORD *)v8 - 4);
  if ( !v9 )
    return 0;
  *a1[1] = v9;
  return result;
}
