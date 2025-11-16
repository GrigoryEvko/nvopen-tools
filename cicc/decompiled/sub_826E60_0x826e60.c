// Function: sub_826E60
// Address: 0x826e60
//
__int64 __fastcall sub_826E60(__int64 *a1, __int64 *a2, int a3, __int64 a4, __int64 a5)
{
  char v7; // dl
  unsigned __int8 v8; // cl
  char v9; // r11
  int v10; // eax
  char v11; // r10
  __int64 result; // rax
  char v13; // r10
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r12
  __int64 v19; // r13

  v7 = *((_BYTE *)a1 + 12);
  v8 = *((_BYTE *)a2 + 12);
  v9 = v8 & 0x20;
  v10 = (((unsigned __int8)v7 >> 5) ^ 1) & 1;
  if ( (_DWORD)qword_4D0495C )
  {
    v11 = v8 & 8;
    if ( (v7 & 8) != 0 )
    {
      result = 0;
      if ( v11 )
        return result;
      v10 = 1;
LABEL_5:
      if ( v9 )
        return 1;
      return (unsigned int)(v10 - 1);
    }
    if ( v11 )
      return (unsigned int)(v10 - 1);
  }
  if ( !v9 )
    return (unsigned int)(v10 - 1);
  if ( (((*((_BYTE *)a1 + 12) >> 5) ^ 1) & 1) != 0 )
    goto LABEL_5;
  v13 = v8 & 0x40;
  if ( (v7 & 0x40) != 0 )
  {
    if ( !v13 )
      return 1;
    if ( ((v7 ^ v8) & 0x80u) == 0 )
      return 0;
    result = 0xFFFFFFFFLL;
    if ( v7 < 0 )
      return 1;
  }
  else
  {
    result = 0xFFFFFFFFLL;
    if ( !v13 )
    {
      if ( ((*((_BYTE *)a1 + 14) ^ *((_BYTE *)a2 + 14)) & 2) != 0 )
      {
        if ( (*((_BYTE *)a1 + 14) & 2) != 0 )
          return 1;
      }
      else if ( dword_4D0439C && ((*((_BYTE *)a1 + 13) ^ *((_BYTE *)a2 + 13)) & 1) != 0 )
      {
        if ( (*((_BYTE *)a1 + 13) & 1) == 0 )
          return 1;
      }
      else
      {
        if ( !dword_4F077BC || ((*((_BYTE *)a1 + 13) ^ *((_BYTE *)a2 + 13)) & 0x40) == 0 )
        {
          v14 = *a1;
          v15 = *a2;
          if ( *a1 && v15 )
          {
            if ( ((v8 ^ (unsigned __int8)v7) & 1) == 0 )
            {
              if ( v14 != v15 )
              {
                if ( (v7 & 1) == (((unsigned __int8)a3 ^ 1) & 1) )
                {
                  if ( sub_8D5CE0(*(_QWORD *)(v15 + 56), *(_QWORD *)(v14 + 56)) )
                    return 1;
                  v16 = *(_QWORD *)(v15 + 56);
                  v17 = *(_QWORD *)(v14 + 56);
                }
                else
                {
                  if ( sub_8D5CE0(*(_QWORD *)(v14 + 40), *(_QWORD *)(v15 + 40)) )
                    return 1;
                  v16 = *(_QWORD *)(v14 + 40);
                  v17 = *(_QWORD *)(v15 + 40);
                }
                return (unsigned int)-(sub_8D5CE0(v17, v16) != 0);
              }
              return 0;
            }
            if ( !a3 )
              return 1;
          }
          else if ( !a3 )
          {
            if ( v14 )
              return 1;
            if ( !v15 )
            {
              if ( (*((_BYTE *)a1 + 13) & 2) == 0
                || (*((_BYTE *)a2 + 13) & 2) == 0
                || (((unsigned __int8)v7 ^ v8) & 0x10) == 0 )
              {
                return 0;
              }
              if ( (*((_BYTE *)a1 + 12) & 0x10) == 0 )
                return 1;
            }
            return 0xFFFFFFFFLL;
          }
          if ( (*((_BYTE *)a1 + 12) & 0x10) != 0 && (*((_BYTE *)a2 + 12) & 0x10) != 0 )
          {
            if ( a4 )
            {
              if ( a5 )
              {
                if ( (unsigned int)sub_8D2E30(a4) )
                {
                  if ( (unsigned int)sub_8D2E30(a5) )
                  {
                    v18 = sub_8D46C0(a4);
                    v19 = sub_8D46C0(a5);
                    if ( (unsigned int)sub_8D3A70(v18) )
                    {
                      if ( (unsigned int)sub_8D3A70(v19) )
                      {
                        if ( sub_8D5CE0(v19, v18) )
                          return 1;
                        v16 = v19;
                        v17 = v18;
                        return (unsigned int)-(sub_8D5CE0(v17, v16) != 0);
                      }
                    }
                  }
                }
              }
            }
          }
          return 0;
        }
        result = 0xFFFFFFFFLL;
        if ( (*((_BYTE *)a1 + 13) & 0x40) == 0 )
          return 1;
      }
    }
  }
  return result;
}
