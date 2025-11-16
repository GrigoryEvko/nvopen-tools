// Function: sub_855880
// Address: 0x855880
//
__int64 sub_855880()
{
  unsigned int v0; // r8d
  const char *v1; // rdx
  const char *v3; // r9
  const char *v4; // r9
  const char *v5; // r9

  if ( qword_4F06400 != 2 )
  {
    if ( qword_4F06400 == 5 )
    {
      v0 = 1;
      v3 = qword_4F06410;
      if ( !memcmp("ifdef", qword_4F06410, 5u) )
        return v0;
      v0 = 7;
      if ( !memcmp("endif", qword_4F06410, 5u) )
        return v0;
      if ( dword_4D04188 )
      {
LABEL_17:
        v0 = 10;
        if ( !memcmp("undef", v3, 5u) )
          return v0;
        v0 = 12;
        if ( !memcmp("error", v3, 5u) )
          return v0;
        if ( !dword_4D04964 )
          return memcmp("ident", v3, 5u) == 0 ? 16 : 22;
        goto LABEL_40;
      }
LABEL_16:
      v3 = qword_4F06410;
      goto LABEL_17;
    }
    if ( qword_4F06400 != 6 )
    {
      if ( qword_4F06400 == 4 )
      {
        v0 = 4;
        v5 = qword_4F06410;
        if ( !memcmp("else", qword_4F06410, 4u) )
          return v0;
        v0 = 3;
        if ( !memcmp("elif", qword_4F06410, 4u) )
          return v0;
        if ( dword_4D04188 )
        {
LABEL_35:
          v0 = 11;
          if ( !memcmp("line", v5, 4u) )
            return v0;
LABEL_9:
          if ( !dword_4D04964 )
            goto LABEL_10;
LABEL_40:
          if ( dword_4F077C4 == 2 )
          {
            if ( unk_4F07778 <= 202301 )
              goto LABEL_42;
          }
          else if ( unk_4F07778 <= 202310 )
          {
LABEL_42:
            if ( qword_4F06400 == 5 )
            {
              v3 = qword_4F06410;
              return memcmp("ident", v3, 5u) == 0 ? 16 : 22;
            }
            if ( qword_4F06400 != 6 )
              goto LABEL_45;
            v4 = qword_4F06410;
            return memcmp("assert", v4, 6u) == 0 ? 17 : 22;
          }
LABEL_10:
          if ( qword_4F06400 == 7 )
            return 21 - ((unsigned int)(memcmp("warning", qword_4F06410, 7u) == 0) - 1);
          goto LABEL_42;
        }
LABEL_34:
        v5 = qword_4F06410;
        goto LABEL_35;
      }
      if ( dword_4D04188 )
        goto LABEL_6;
      goto LABEL_37;
    }
    v0 = 2;
    if ( !memcmp("ifndef", qword_4F06410, 6u) )
      return v0;
LABEL_22:
    if ( dword_4D04188 )
    {
LABEL_6:
      switch ( qword_4F06400 )
      {
        case 7uLL:
          v0 = 5;
          v1 = qword_4F06410;
          if ( !memcmp("elifdef", qword_4F06410, 7u) )
            return v0;
          goto LABEL_8;
        case 8uLL:
          v0 = 6;
          if ( !memcmp("elifndef", qword_4F06410, 8u) )
            return v0;
          if ( !dword_4D04964 )
            return memcmp("unassert", qword_4F06410, 8u) == 0 ? 18 : 22;
          goto LABEL_40;
        case 6uLL:
LABEL_24:
          v4 = qword_4F06410;
          v0 = 9;
          if ( !memcmp("define", qword_4F06410, 6u) )
            return v0;
          v0 = 13;
          if ( !memcmp("pragma", qword_4F06410, 6u) )
            return v0;
          if ( !dword_4D04964 )
            return memcmp("assert", v4, 6u) == 0 ? 17 : 22;
          goto LABEL_40;
      }
LABEL_56:
      if ( qword_4F06400 != 5 )
      {
        if ( qword_4F06400 != 4 )
        {
          if ( !dword_4D04964 )
          {
LABEL_45:
            if ( qword_4F06400 != 8 )
            {
              if ( qword_4F06400 == 12 )
                return memcmp("include_next", qword_4F06410, 0xCu) == 0 ? 20 : 22;
              else
                return 22;
            }
            return memcmp("unassert", qword_4F06410, 8u) == 0 ? 18 : 22;
          }
          goto LABEL_40;
        }
        goto LABEL_34;
      }
      goto LABEL_16;
    }
    if ( qword_4F06400 == 6 )
      goto LABEL_24;
LABEL_37:
    if ( qword_4F06400 == 7 )
    {
      v1 = qword_4F06410;
LABEL_8:
      v0 = 8;
      if ( memcmp("include", v1, 7u) )
        goto LABEL_9;
      return v0;
    }
    goto LABEL_56;
  }
  if ( *qword_4F06410 != 105 || qword_4F06410[1] != 102 )
    goto LABEL_22;
  return 0;
}
