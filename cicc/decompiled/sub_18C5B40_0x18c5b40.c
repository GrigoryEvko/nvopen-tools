// Function: sub_18C5B40
// Address: 0x18c5b40
//
__int64 __fastcall sub_18C5B40(__int64 a1)
{
  unsigned __int8 *v1; // rcx
  __int64 result; // rax

  v1 = *(unsigned __int8 **)(a1 + 8 * (1LL - *(unsigned int *)(a1 + 8)));
  result = 0;
  if ( (unsigned int)*v1 - 1 <= 1 )
  {
    result = *((_QWORD *)v1 + 17);
    if ( result )
    {
      if ( *(_BYTE *)(result + 16) == 13 )
      {
        if ( *(_DWORD *)(result + 32) != 64 )
          return 0;
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
