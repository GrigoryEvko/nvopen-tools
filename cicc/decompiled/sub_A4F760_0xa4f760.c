// Function: sub_A4F760
// Address: 0xa4f760
//
__int64 __fastcall sub_A4F760(unsigned __int8 *a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // r12
  __int64 v4; // rax

  v1 = *a1;
  if ( *a1 == 22 )
  {
    v2 = *((_QWORD *)a1 + 3);
    if ( v2 )
      return *(_QWORD *)(v2 + 40);
  }
  else
  {
    if ( v1 == 23 )
    {
      v2 = *((_QWORD *)a1 + 9);
      if ( !v2 )
        return v2;
      return *(_QWORD *)(v2 + 40);
    }
    if ( v1 <= 0x1Cu )
    {
      if ( v1 > 3u )
      {
        v2 = 0;
        if ( v1 == 24 )
        {
          v2 = *((_QWORD *)a1 + 2);
          if ( v2 )
          {
            while ( 1 )
            {
              if ( **(_BYTE **)(v2 + 24) > 0x1Cu )
              {
                v4 = sub_A4F760();
                if ( v4 )
                  break;
              }
              v2 = *(_QWORD *)(v2 + 8);
              if ( !v2 )
                return v2;
            }
            return v4;
          }
        }
      }
      else
      {
        return *((_QWORD *)a1 + 5);
      }
    }
    else
    {
      v2 = *((_QWORD *)a1 + 5);
      if ( v2 )
      {
        v2 = *(_QWORD *)(v2 + 72);
        if ( v2 )
          return *(_QWORD *)(v2 + 40);
      }
    }
  }
  return v2;
}
