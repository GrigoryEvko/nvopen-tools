// Function: sub_10C8AB0
// Address: 0x10c8ab0
//
__int64 __fastcall sub_10C8AB0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  _BYTE *v6; // r13
  _BYTE *v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rdx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  if ( !v4 )
    goto LABEL_16;
  **a1 = v4;
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( *v6 != 59 )
    goto LABEL_5;
  result = sub_995B10(a1 + 1, *((_QWORD *)v6 - 8));
  v8 = *((_QWORD *)v6 - 4);
  if ( (_BYTE)result && v8 )
    goto LABEL_9;
  result = sub_995B10(a1 + 1, v8);
  if ( !(_BYTE)result || (v9 = *((_QWORD *)v6 - 8)) == 0 )
  {
LABEL_16:
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
    if ( !v6 )
      return 0;
LABEL_5:
    **a1 = v6;
    v7 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( *v7 != 59 )
      return 0;
    result = sub_995B10(a1 + 1, *((_QWORD *)v7 - 8));
    v8 = *((_QWORD *)v7 - 4);
    if ( !(_BYTE)result || !v8 )
    {
      result = sub_995B10(a1 + 1, v8);
      if ( !(_BYTE)result )
        return 0;
      v9 = *((_QWORD *)v7 - 8);
      if ( !v9 )
        return 0;
      goto LABEL_13;
    }
LABEL_9:
    *a1[2] = v8;
    return result;
  }
LABEL_13:
  *a1[2] = v9;
  return result;
}
