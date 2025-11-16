// Function: sub_B93270
// Address: 0xb93270
//
__int64 __fastcall sub_B93270(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  if ( a2
    && (result = (unsigned int)*a2 - 5, (unsigned __int8)(*a2 - 5) <= 0x1Fu)
    && ((result = a2[1] & 0x7F, (_BYTE)result == 2) || (a4 = *((unsigned int *)a2 - 2), (_DWORD)a4)) )
  {
    if ( !a3 )
      return sub_B93250(a1, (__int64)a2, (__int64)a3, a4, a5);
    if ( (unsigned __int8)(*a3 - 5) > 0x1Fu )
      return sub_B93250(a1, (__int64)a2, (__int64)a3, a4, a5);
    result = a3[1] & 0x7F;
    if ( (_BYTE)result != 2 )
    {
      result = *((unsigned int *)a3 - 2);
      if ( !(_DWORD)result )
        return sub_B93250(a1, (__int64)a2, (__int64)a3, a4, a5);
    }
  }
  else if ( a3 )
  {
    result = (unsigned int)*a3 - 5;
    if ( (unsigned __int8)(*a3 - 5) <= 0x1Fu )
    {
      result = a3[1] & 0x7F;
      if ( (_BYTE)result == 2 || *((_DWORD *)a3 - 2) )
        ++*(_DWORD *)(a1 - 8);
    }
  }
  return result;
}
