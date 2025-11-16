// Function: sub_3074500
// Address: 0x3074500
//
bool __fastcall sub_3074500(__int64 a1, char *a2, __int64 a3, int a4)
{
  bool result; // al
  __int64 v5; // rdx
  __int64 v6; // rdx
  unsigned int v7; // edi
  char v8; // cl
  __int64 v9; // rdi

  if ( a4 == 4 )
  {
    v8 = *a2;
    if ( *a2 == 62 )
    {
      return *((_QWORD *)a2 - 4) != a3;
    }
    else
    {
      result = 0;
      if ( (unsigned __int8)(v8 - 65) > 1u )
      {
        result = 1;
        if ( v8 == 85 )
        {
          v9 = *((_QWORD *)a2 - 4);
          if ( v9 )
          {
            if ( !*(_BYTE *)v9
              && *(_QWORD *)(v9 + 24) == *((_QWORD *)a2 + 10)
              && (*(_BYTE *)(v9 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v9 + 36) - 238) <= 7
              && ((1LL << (*(_BYTE *)(v9 + 36) + 18)) & 0xAD) != 0 )
            {
              return *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] != a3;
            }
            else
            {
              result = 1;
              if ( !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *((_QWORD *)a2 + 10) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
              {
                v7 = *(_DWORD *)(v9 + 36);
                switch ( v7 )
                {
                  case 0x1FF5u:
                  case 0x1FF6u:
                  case 0x1FF7u:
                  case 0x1FF8u:
                  case 0x1FFFu:
                  case 0x2007u:
                  case 0x2008u:
                  case 0x200Fu:
                  case 0x2010u:
                  case 0x2011u:
                  case 0x2012u:
                  case 0x2013u:
                  case 0x2014u:
                  case 0x2017u:
                  case 0x2018u:
LABEL_29:
                    result = 0;
                    break;
                  default:
LABEL_30:
                    result = !sub_3071900(v7);
                    break;
                }
              }
            }
          }
        }
      }
    }
  }
  else if ( a4 == 5 )
  {
    result = 0;
    if ( (unsigned __int8)(*a2 - 65) > 1u )
    {
      result = 1;
      if ( *a2 == 85 )
      {
        v6 = *((_QWORD *)a2 - 4);
        if ( v6 )
        {
          if ( !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *((_QWORD *)a2 + 10) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
          {
            v7 = *(_DWORD *)(v6 + 36);
            switch ( v7 )
            {
              case 0x1FF5u:
              case 0x1FF6u:
              case 0x1FF7u:
              case 0x1FF8u:
              case 0x1FFFu:
              case 0x2007u:
              case 0x2008u:
              case 0x200Fu:
              case 0x2010u:
              case 0x2011u:
              case 0x2012u:
              case 0x2013u:
              case 0x2014u:
              case 0x2017u:
              case 0x2018u:
                goto LABEL_29;
              default:
                goto LABEL_30;
            }
          }
        }
      }
    }
  }
  else
  {
    result = 1;
    if ( a4 == 3 && *a2 == 85 )
    {
      v5 = *((_QWORD *)a2 - 4);
      if ( v5 )
      {
        if ( !*(_BYTE *)v5 && *(_QWORD *)(v5 + 24) == *((_QWORD *)a2 + 10) && (*(_BYTE *)(v5 + 33) & 0x20) != 0 )
          return (unsigned int)(*(_DWORD *)(v5 + 36) - 8215) > 1;
      }
    }
  }
  return result;
}
