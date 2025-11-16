// Function: sub_1FF15D0
// Address: 0x1ff15d0
//
__int64 __fastcall sub_1FF15D0(char *a1, _QWORD *a2)
{
  __int64 v2; // rbx
  unsigned int v4; // esi
  int v5; // ecx
  int v6; // ecx
  unsigned int v7; // esi
  __int64 v8; // rax

  if ( *a1 )
    v4 = sub_1FEB8F0(*a1);
  else
    v4 = sub_1F58D40((__int64)a1);
  LOBYTE(v5) = 2;
  while ( 2 * (unsigned int)sub_1FEB8F0(v5) < v4 )
  {
    v5 = v6 + 1;
    if ( v5 == 8 )
    {
      v7 = (v4 + 1) >> 1;
      if ( v7 == 32 )
      {
        LOBYTE(v8) = 5;
      }
      else if ( v7 > 0x20 )
      {
        if ( v7 == 64 )
        {
          LOBYTE(v8) = 6;
        }
        else
        {
          if ( v7 != 128 )
            goto LABEL_15;
          LOBYTE(v8) = 7;
        }
      }
      else if ( v7 == 8 )
      {
        LOBYTE(v8) = 3;
      }
      else
      {
        LOBYTE(v8) = 4;
        if ( v7 != 16 )
        {
          LOBYTE(v8) = 2;
          if ( v7 != 1 )
          {
LABEL_15:
            v8 = sub_1F58CC0(a2, v7);
            v2 = v8;
          }
        }
      }
      LOBYTE(v2) = v8;
      return v2;
    }
  }
  return (unsigned __int8)v6;
}
