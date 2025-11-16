// Function: sub_10E4850
// Address: 0x10e4850
//
_BOOL8 __fastcall sub_10E4850(_QWORD **a1, unsigned __int8 *a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rcx
  int v5; // edx
  _BOOL4 v6; // r12d
  _BYTE *v7; // rdi
  __int64 v9; // rax

  v3 = *a2;
  if ( (unsigned __int8)v3 <= 0x1Cu )
  {
    if ( (_BYTE)v3 != 5 )
      return 0;
    v5 = *((unsigned __int16 *)a2 + 1);
    if ( (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD && (v5 & 0xFFF7) != 0x11 )
      return 0;
  }
  else
  {
    if ( (unsigned __int8)v3 > 0x36u )
      return 0;
    v4 = 0x40540000000000LL;
    v5 = (unsigned __int8)v3 - 29;
    if ( !_bittest64(&v4, v3) )
      return 0;
  }
  if ( v5 == 15 )
  {
    v6 = (a2[1] & 2) != 0;
    if ( (a2[1] & 2) != 0 )
    {
      v7 = (_BYTE *)*((_QWORD *)a2 - 8);
      if ( *v7 <= 0x15u )
      {
        **a1 = v7;
        if ( *v7 > 0x15u || *v7 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v7) )
        {
          v9 = *((_QWORD *)a2 - 4);
          if ( v9 )
          {
            *a1[2] = v9;
            return v6;
          }
        }
      }
    }
  }
  return 0;
}
