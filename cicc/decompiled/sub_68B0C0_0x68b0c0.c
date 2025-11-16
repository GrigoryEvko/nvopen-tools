// Function: sub_68B0C0
// Address: 0x68b0c0
//
_DWORD *__fastcall sub_68B0C0(_DWORD *a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  _DWORD *result; // rax

  result = &dword_4F077C4;
  if ( dword_4F077C4 != 2 )
  {
    if ( (*((_BYTE *)a1 + 161) & 8) != 0 )
    {
      result = a1;
    }
    else
    {
      result = (_DWORD *)*((_QWORD *)a1 + 21);
      if ( !result )
        return result;
    }
    if ( (*(_BYTE *)(a2 + 161) & 8) != 0 )
    {
      a1 = result;
      if ( *((_BYTE *)result + 140) != 2 )
        return result;
      goto LABEL_12;
    }
    if ( *(_QWORD *)(a2 + 168) )
    {
      a2 = *(_QWORD *)(a2 + 168);
      a1 = result;
    }
  }
  if ( *((_BYTE *)a1 + 140) != 2 )
    return result;
LABEL_12:
  if ( (*((_BYTE *)a1 + 161) & 8) != 0
    && *(_BYTE *)(a2 + 140) == 2
    && (*(_BYTE *)(a2 + 161) & 8) != 0
    && a1 != (_DWORD *)a2 )
  {
    if ( !dword_4F07588 )
      return (_DWORD *)sub_6E5D70(a4, 2551, a3, a1, a2);
    result = (_DWORD *)*((_QWORD *)a1 + 4);
    if ( *(_DWORD **)(a2 + 32) != result || !result )
      return (_DWORD *)sub_6E5D70(a4, 2551, a3, a1, a2);
  }
  return result;
}
