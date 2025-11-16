// Function: sub_161F190
// Address: 0x161f190
//
__int64 __fastcall sub_161F190(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 result; // rax

  if ( a2
    && (result = (unsigned int)*a2 - 4, (unsigned __int8)(*a2 - 4) <= 0x1Eu)
    && (a2[1] == 2 || *((_DWORD *)a2 + 3)) )
  {
    if ( !a3 )
      return sub_161EA80(a1);
    result = (unsigned int)*a3 - 4;
    if ( (unsigned __int8)(*a3 - 4) > 0x1Eu )
      return sub_161EA80(a1);
    if ( a3[1] != 2 )
    {
      result = *((unsigned int *)a3 + 3);
      if ( !(_DWORD)result )
        return sub_161EA80(a1);
    }
  }
  else if ( a3 )
  {
    result = (unsigned int)*a3 - 4;
    if ( (unsigned __int8)(*a3 - 4) <= 0x1Eu && (a3[1] == 2 || *((_DWORD *)a3 + 3)) )
      ++*(_DWORD *)(a1 + 12);
  }
  return result;
}
