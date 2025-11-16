// Function: sub_10E2620
// Address: 0x10e2620
//
__int64 __fastcall sub_10E2620(_QWORD **a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rax

  v2 = *a2;
  if ( (unsigned __int8)v2 <= 0x1Cu )
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v4 = *((unsigned __int16 *)a2 + 1);
    if ( (*((_WORD *)a2 + 1) & 0xFFF7) != 0x11 && (v4 & 0xFFFD) != 0xD )
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
  if ( v4 == 15 && (a2[1] & 4) != 0 )
  {
    v5 = *((_QWORD *)a2 - 8);
    if ( v5 )
    {
      **a1 = v5;
      v6 = *((_QWORD *)a2 - 4);
      if ( v6 )
      {
        *a1[1] = v6;
        return 1;
      }
    }
  }
  return 0;
}
