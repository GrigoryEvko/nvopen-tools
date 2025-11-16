// Function: sub_2535530
// Address: 0x2535530
//
__int64 __fastcall sub_2535530(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= a2 )
    a2 = *(_DWORD *)(a1 + 12);
  if ( a2 < (unsigned int)result )
    a2 = *(_DWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 12) = a2;
  return result;
}
