// Function: sub_CC5470
// Address: 0xcc5470
//
__int64 __fastcall sub_CC5470(_DWORD *a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 result; // rax
  _QWORD *v6; // rdx
  _DWORD *v7; // rdx
  _DWORD *v8; // rdx

  if ( a2 <= 3 )
  {
LABEL_2:
    if ( a2 != 4 )
    {
      v2 = sub_F05A00(a1);
      if ( v3 )
        goto LABEL_4;
      return 0;
    }
    goto LABEL_14;
  }
  if ( *a1 == 1936746861 && (*(_DWORD *)((char *)a1 + a2 - 4) == 1818572402 || *(_WORD *)((char *)a1 + a2 - 2) == 13938) )
    return 40;
  switch ( a2 )
  {
    case 0xAuLL:
      if ( *(_QWORD *)a1 == 0x7363707265776F70LL )
      {
        result = 41;
        if ( *((_WORD *)a1 + 4) == 25968 )
          return result;
      }
      break;
    case 6uLL:
      if ( *a1 == 913142369 )
      {
        result = 35;
        if ( *((_WORD *)a1 + 2) == 25908 )
          return result;
      }
      break;
    case 7uLL:
      if ( *a1 == 913142369 && *((_WORD *)a1 + 2) == 25908 )
      {
        result = 36;
        if ( *((_BYTE *)a1 + 6) == 99 )
          return result;
      }
      break;
    default:
      if ( a2 <= 4 )
        goto LABEL_2;
      break;
  }
  if ( *a1 == 1919512691 && *((_BYTE *)a1 + 4) == 118 )
  {
    v8 = (_DWORD *)((char *)a1 + a2 - 4);
    result = 42;
    if ( *v8 != 808333686 )
    {
      result = 43;
      if ( *v8 != 825110902 )
      {
        result = 44;
        if ( *v8 != 841888118 )
        {
          result = 45;
          if ( *v8 != 858665334 )
          {
            result = 46;
            if ( *v8 != 875442550 )
            {
              result = 47;
              if ( *v8 != 892219766 )
              {
                result = 0;
                if ( *v8 == 908996982 )
                  return 48;
              }
            }
          }
        }
      }
    }
    return result;
  }
LABEL_14:
  if ( *a1 == 1818851428 )
  {
    v7 = (_DWORD *)((char *)a1 + a2 - 4);
    result = 49;
    if ( *v7 != 808333686 )
    {
      result = 50;
      if ( *v7 != 825110902 )
      {
        result = 51;
        if ( *v7 != 841888118 )
        {
          result = 52;
          if ( *v7 != 858665334 )
          {
            result = 53;
            if ( *v7 != 875442550 )
            {
              result = 54;
              if ( *v7 != 892219766 )
              {
                result = 55;
                if ( *v7 != 908996982 )
                {
                  result = 56;
                  if ( *v7 != 925774198 )
                  {
                    result = 0;
                    if ( *v7 == 942551414 )
                      return 57;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v2 = sub_F05A00(a1);
    if ( v3 )
    {
LABEL_4:
      v4 = (unsigned int)sub_F05240(v2, v3) - 2;
      result = 0;
      if ( (unsigned int)v4 <= 0x27 )
        return byte_3F6C320[v4];
      return result;
    }
    if ( a2 <= 7 )
      return 0;
    v6 = (_QWORD *)((char *)a1 + a2 - 8);
    if ( *v6 == 0x3361626D696C616BLL )
      return 37;
    result = 38;
    if ( *v6 != 0x3461626D696C616BLL )
    {
      if ( *v6 == 0x3561626D696C616BLL )
        return 39;
      return 0;
    }
  }
  return result;
}
