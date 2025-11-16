// Function: sub_8C39E0
// Address: 0x8c39e0
//
__int64 *__fastcall sub_8C39E0(__int64 *a1, unsigned __int8 a2)
{
  __int64 *result; // rax
  __int64 v3; // rcx

  result = a1;
  if ( a1 && (*(_BYTE *)(a1 - 1) & 2) != 0 )
  {
    v3 = *(a1 - 3);
    if ( v3 )
    {
      result = (__int64 *)*(a1 - 3);
      if ( (*(_BYTE *)(v3 - 8) & 2) != 0 )
        return *(__int64 **)(v3 - 24);
    }
    else if ( (*(_BYTE *)(a1 - 1) & 1) != 0 )
    {
      sub_8C3650(a1, a2, 0);
      result = (__int64 *)*(a1 - 3);
      if ( (*(_BYTE *)(result - 1) & 2) != 0 )
        return (__int64 *)*(result - 3);
    }
  }
  return result;
}
