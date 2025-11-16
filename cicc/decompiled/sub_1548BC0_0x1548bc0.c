// Function: sub_1548BC0
// Address: 0x1548bc0
//
_QWORD *__fastcall sub_1548BC0(__int64 a1)
{
  unsigned __int8 v1; // al
  _QWORD *v2; // r12
  __int64 v4; // rax
  __int64 v5; // rax

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 17 )
  {
    v2 = *(_QWORD **)(a1 + 24);
    if ( v2 )
      return (_QWORD *)v2[5];
  }
  else
  {
    if ( v1 == 18 )
    {
      v2 = *(_QWORD **)(a1 + 56);
      if ( !v2 )
        return v2;
      return (_QWORD *)v2[5];
    }
    if ( v1 <= 0x17u )
    {
      if ( v1 > 3u )
      {
        v2 = 0;
        if ( v1 == 19 )
        {
          v2 = *(_QWORD **)(a1 + 8);
          if ( v2 )
          {
            while ( 1 )
            {
              v4 = sub_1648700(v2);
              if ( *(_BYTE *)(v4 + 16) > 0x17u )
              {
                v5 = sub_1548BC0(v4);
                if ( v5 )
                  break;
              }
              v2 = (_QWORD *)v2[1];
              if ( !v2 )
                return v2;
            }
            return (_QWORD *)v5;
          }
        }
      }
      else
      {
        return *(_QWORD **)(a1 + 40);
      }
    }
    else
    {
      v2 = *(_QWORD **)(a1 + 40);
      if ( v2 )
      {
        v2 = (_QWORD *)v2[7];
        if ( v2 )
          return (_QWORD *)v2[5];
      }
    }
  }
  return v2;
}
