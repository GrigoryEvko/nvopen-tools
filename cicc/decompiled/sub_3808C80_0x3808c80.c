// Function: sub_3808C80
// Address: 0x3808c80
//
unsigned __int8 *__fastcall sub_3808C80(__int64 a1, unsigned __int64 a2)
{
  unsigned __int8 *v2; // rax
  int v4; // edx
  __int16 v5; // ax

  v2 = sub_3452270(*(_QWORD *)a1, a2, *(_QWORD **)(a1 + 8));
  if ( v2 )
    return sub_3808AD0(a1, (__int64)v2);
  v4 = 267;
  v5 = **(_WORD **)(a2 + 48);
  if ( v5 != 12 )
  {
    v4 = 268;
    if ( v5 != 13 )
    {
      v4 = 269;
      if ( v5 != 14 )
      {
        v4 = 270;
        if ( v5 != 15 )
        {
          v4 = 729;
          if ( v5 == 16 )
            v4 = 271;
        }
      }
    }
  }
  return (unsigned __int8 *)sub_38062F0((__int64 *)a1, a2, v4);
}
