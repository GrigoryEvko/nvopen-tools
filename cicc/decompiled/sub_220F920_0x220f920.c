// Function: sub_220F920
// Address: 0x220f920
//
__int64 __fastcall sub_220F920(__int64 a1, unsigned __int64 a2)
{
  unsigned __int8 *v2; // rax
  int v3; // r9d
  unsigned __int64 v4; // rdx
  int v5; // ecx
  int v7; // edx
  int v8; // edx

  v2 = *(unsigned __int8 **)a1;
  v3 = -2;
  v4 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
  if ( !v4 )
    return (unsigned int)v3;
  v5 = *v2;
  if ( (v5 & 0x80u) != 0 )
  {
    v3 = -1;
    if ( (unsigned __int8)v5 > 0xC1u )
    {
      if ( (unsigned __int8)v5 > 0xDFu )
      {
        if ( (unsigned __int8)v5 > 0xEFu )
        {
          if ( (unsigned __int8)v5 <= 0xF4u )
          {
            v3 = -2;
            if ( v4 > 3 )
            {
              v8 = v2[1];
              v3 = -1;
              if ( (v2[1] & 0xC0) == 0x80 && ((_BYTE)v5 != 0xF0 || (unsigned __int8)v8 > 0x8Fu) )
              {
                if ( (_BYTE)v5 != 0xF4 || (v3 = -1, (unsigned __int8)v8 <= 0x8Fu) )
                {
                  v3 = -1;
                  if ( (v2[2] & 0xC0) == 0x80 && (v2[3] & 0xC0) == 0x80 )
                  {
                    v3 = (v8 << 12) + (v5 << 18) + (v2[2] << 6) + v2[3] - 63447168;
                    if ( v3 <= a2 )
                      *(_QWORD *)a1 = v2 + 4;
                  }
                }
              }
            }
          }
        }
        else
        {
          v3 = -2;
          if ( v4 > 2 )
          {
            v7 = v2[1];
            v3 = -1;
            if ( (v2[1] & 0xC0) == 0x80 && ((_BYTE)v5 != 0xE0 || (unsigned __int8)v7 > 0x9Fu) )
            {
              v3 = -1;
              if ( (v2[2] & 0xC0) == 0x80 )
              {
                v3 = (v5 << 12) + (v7 << 6) + v2[2] - 925824;
                if ( v3 <= a2 )
                  *(_QWORD *)a1 = v2 + 3;
              }
            }
          }
        }
      }
      else if ( v4 == 1 )
      {
        return (unsigned int)-2;
      }
      else if ( (v2[1] & 0xC0) == 0x80 )
      {
        v3 = (v5 << 6) + v2[1] - 12416;
        if ( v3 <= a2 )
          *(_QWORD *)a1 = v2 + 2;
      }
    }
    return (unsigned int)v3;
  }
  *(_QWORD *)a1 = v2 + 1;
  return (unsigned __int8)v5;
}
