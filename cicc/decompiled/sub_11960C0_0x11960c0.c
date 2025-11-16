// Function: sub_11960C0
// Address: 0x11960c0
//
__int64 __fastcall sub_11960C0(_QWORD **a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rcx

  v2 = *((_QWORD *)a2 + 2);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v4 = *a2;
  if ( (unsigned __int8)v4 > 0x1Cu )
  {
    if ( (unsigned __int8)v4 > 0x36u )
      return 0;
    v8 = 0x40540000000000LL;
    if ( !_bittest64(&v8, v4) )
      return 0;
    v5 = (unsigned __int8)v4 - 29;
  }
  else
  {
    if ( (_BYTE)v4 != 5 )
      return 0;
    v5 = *((unsigned __int16 *)a2 + 1);
    if ( (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD && (v5 & 0xFFF7) != 0x11 )
      return 0;
  }
  if ( v5 != 15 )
    return 0;
  if ( (a2[1] & 4) == 0 )
    return 0;
  v6 = *((_QWORD *)a2 - 8);
  if ( !v6 )
    return 0;
  **a1 = v6;
  v7 = *((_QWORD *)a2 - 4);
  if ( !v7 )
    return 0;
  *a1[1] = v7;
  return 1;
}
