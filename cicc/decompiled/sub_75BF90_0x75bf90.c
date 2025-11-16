// Function: sub_75BF90
// Address: 0x75bf90
//
__int64 *__fastcall sub_75BF90(__int64 a1)
{
  __int64 *result; // rax
  __int64 v3; // rdx
  __int64 *v4; // rdx
  __int64 v5; // rdx

  result = (__int64 *)(unsigned int)dword_4F08010;
  while ( 1 )
  {
    while ( (_DWORD)result && (*(_BYTE *)(a1 - 8) & 2) == 0 )
    {
      v4 = *(__int64 **)(a1 + 32);
      if ( !v4 )
        return result;
      v5 = *v4;
      if ( a1 == v5 || (*(_BYTE *)(v5 - 8) & 2) == 0 )
        return result;
      a1 = v5;
    }
    if ( *(char *)(a1 + 178) < 0 )
      return result;
    *(_BYTE *)(a1 + 178) |= 0x80u;
    if ( *(char *)(a1 - 8) < 0 )
    {
      sub_750670(a1, 6);
      sub_75B260(a1, 6u);
      result = *(__int64 **)(a1 + 32);
      if ( !result )
        return result;
    }
    else
    {
      result = *(__int64 **)(a1 + 32);
      if ( !result )
        return result;
    }
    v3 = *result;
    if ( a1 == *result || (*(_BYTE *)(v3 - 8) & 2) == 0 )
      break;
    result = (__int64 *)(unsigned int)dword_4F08010;
    a1 = v3;
  }
  return result;
}
