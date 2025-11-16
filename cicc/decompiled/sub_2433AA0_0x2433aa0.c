// Function: sub_2433AA0
// Address: 0x2433aa0
//
__int64 __fastcall sub_2433AA0(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 156) )
    result = *(unsigned int *)(a1 + 152);
  *(_DWORD *)(a1 + 136) = result;
  return result;
}
