// Function: sub_737310
// Address: 0x737310
//
char *__fastcall sub_737310(__int64 a1, char a2)
{
  char *result; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  char *v6; // rdi
  char *v7; // rdi

  result = (char *)sub_7217A0();
  if ( !result )
  {
    v4 = qword_4F07A28;
    if ( !qword_4F07A28 )
    {
      result = *(char **)(a1 + 40);
      if ( result == (char *)unk_4F07288 )
        goto LABEL_42;
      if ( (*(_BYTE *)(a1 + 89) & 4) == 0 && (!result || result[28] != 3) )
        return result;
      result = (char *)sub_736990(a1);
      if ( !(_DWORD)result )
      {
LABEL_42:
        result = (char *)sub_729F80(*(_DWORD *)(a1 + 64));
        if ( !(_DWORD)result )
        {
          if ( a2 == 7 )
          {
            if ( !*(_BYTE *)(a1 + 136)
              && *(_BYTE *)(a1 + 177)
              && (*(_BYTE *)(a1 + 170) & 0x10) == 0
              && (*(_BYTE *)(a1 + 172) & 0x20) == 0
              && *(char *)(a1 + 156) >= 0
              && (*(_BYTE *)(a1 + 168) & 8) == 0
              && (*(_BYTE *)(a1 + 175) & 1) == 0 )
            {
              qword_4F07A28 = a1;
              v4 = a1;
              byte_4F07A20 = 7;
              goto LABEL_16;
            }
          }
          else if ( a2 == 11
                 && !*(_BYTE *)(a1 + 172)
                 && *(char *)(a1 + 192) >= 0
                 && (*(_BYTE *)(a1 + 195) & 1) == 0
                 && (*(_BYTE *)(a1 + 196) & 0x1C) == 0
                 && (*(_BYTE *)(a1 + 200) & 0x20) == 0 )
          {
            result = (char *)sub_8D96E0(*(_QWORD *)(a1 + 152));
            if ( !(_DWORD)result )
            {
              result = (char *)sub_80CEF0(a1);
              if ( !(_DWORD)result )
              {
                result = (char *)sub_8D3EA0(*(_QWORD *)(*(_QWORD *)(a1 + 152) + 160LL));
                if ( !(_DWORD)result )
                {
                  qword_4F07A28 = a1;
                  v4 = a1;
                  byte_4F07A20 = 11;
LABEL_5:
                  if ( dword_4F077C4 == 2 )
                  {
                    v7 = (char *)sub_815600();
                    return sub_723F40(v7);
                  }
                  return sub_723F40(*(char **)(v4 + 8));
                }
              }
            }
          }
        }
      }
      v4 = qword_4F07A28;
      if ( !qword_4F07A28 )
        return result;
    }
    if ( byte_4F07A20 != 7 )
      goto LABEL_5;
LABEL_16:
    if ( (*(_BYTE *)(v4 + 89) & 4) != 0 || (v5 = *(_QWORD *)(v4 + 40)) != 0 && *(_BYTE *)(v5 + 28) == 3 )
    {
      v6 = (char *)sub_813030();
      return sub_723F40(v6);
    }
    return sub_723F40(*(char **)(v4 + 8));
  }
  return result;
}
