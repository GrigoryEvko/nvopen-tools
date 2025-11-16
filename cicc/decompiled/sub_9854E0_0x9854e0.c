// Function: sub_9854E0
// Address: 0x9854e0
//
__int64 __fastcall sub_9854E0(unsigned __int8 *a1, __int64 (__fastcall *a2)(__int64, unsigned __int8 *), __int64 a3)
{
  __int64 result; // rax
  unsigned __int8 *v6; // rsi
  unsigned __int8 *v7; // rcx

  result = *a1;
  if ( (_BYTE)result == 22 || (unsigned __int8)result <= 3u )
  {
    v6 = a1;
    return a2(a3, v6);
  }
  if ( (unsigned __int8)result > 0x1Cu )
  {
    a2(a3, a1);
    result = *a1;
    if ( (unsigned __int8)result <= 0x1Cu )
    {
      if ( *((_WORD *)a1 + 1) != 47 )
        return result;
    }
    else if ( (_BYTE)result != 76 )
    {
      goto LABEL_8;
    }
    if ( (a1[7] & 0x40) != 0 )
      v7 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
    else
      v7 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v6 = *(unsigned __int8 **)v7;
    if ( *(_QWORD *)v7 )
    {
LABEL_10:
      result = *v6;
      if ( (unsigned __int8)result > 0x1Cu || (_BYTE)result == 22 )
        return a2(a3, v6);
      return result;
    }
LABEL_8:
    if ( (_BYTE)result != 67 )
      return result;
    v6 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
    if ( !v6 )
      return result;
    goto LABEL_10;
  }
  return result;
}
