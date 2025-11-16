// Function: sub_10D7280
// Address: 0x10d7280
//
__int64 __fastcall sub_10D7280(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  _BYTE *v5; // r13
  __int64 v6; // rax
  _BYTE *v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned __int8 *v11; // [rsp-30h] [rbp-30h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  if ( !v4 )
    goto LABEL_6;
  **a1 = v4;
  v5 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( !v5 )
    return 0;
  *a1[1] = v5;
  if ( *v5 != 59 )
  {
LABEL_6:
    v6 = *((_QWORD *)a3 - 4);
    if ( v6 )
    {
      **a1 = v6;
      v7 = (_BYTE *)*((_QWORD *)a3 - 8);
      if ( v7 )
      {
        *a1[1] = v7;
        if ( *v7 == 59 )
        {
          result = sub_995B10(a1 + 2, *((_QWORD *)v7 - 8));
          v8 = *((_QWORD *)v7 - 4);
          if ( !(_BYTE)result || !v8 )
          {
            result = sub_995B10(a1 + 2, v8);
            if ( (_BYTE)result )
            {
              v9 = *((_QWORD *)v7 - 8);
              if ( v9 )
              {
                *a1[3] = v9;
                return result;
              }
            }
            return 0;
          }
          goto LABEL_19;
        }
      }
    }
    return 0;
  }
  v11 = a3;
  result = sub_995B10(a1 + 2, *((_QWORD *)v5 - 8));
  v8 = *((_QWORD *)v5 - 4);
  if ( !(_BYTE)result || !v8 )
  {
    result = sub_995B10(a1 + 2, v8);
    a3 = v11;
    if ( (_BYTE)result )
    {
      v10 = *((_QWORD *)v5 - 8);
      if ( v10 )
      {
        *a1[3] = v10;
        return result;
      }
    }
    goto LABEL_6;
  }
LABEL_19:
  *a1[3] = v8;
  return result;
}
