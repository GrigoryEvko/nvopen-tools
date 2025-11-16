// Function: sub_722A20
// Address: 0x722a20
//
__int64 __fastcall sub_722A20(unsigned __int64 a1, _BYTE *a2)
{
  unsigned __int64 v3; // rax
  char v4; // dl
  unsigned __int64 v5; // rcx
  char v6; // al

  if ( a1 > 0x7F )
  {
    v3 = a1 >> 6;
    v4 = a1 & 0x3F | 0x80;
    if ( a1 <= 0x7FF )
    {
      a2[1] = v4;
      *a2 = v3 | 0xC0;
      return 2;
    }
    else
    {
      v5 = a1 >> 12;
      v6 = v3 & 0x3F | 0x80;
      if ( a1 > 0xFFFF )
      {
        a2[2] = v6;
        a2[3] = v4;
        a2[1] = v5 & 0x3F | 0x80;
        *a2 = (a1 >> 18) & 7 | 0xF0;
        return 4;
      }
      else
      {
        a2[1] = v6;
        *a2 = v5 | 0xE0;
        a2[2] = v4;
        return 3;
      }
    }
  }
  else
  {
    *a2 = a1;
    return 1;
  }
}
