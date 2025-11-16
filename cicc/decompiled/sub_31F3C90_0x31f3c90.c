// Function: sub_31F3C90
// Address: 0x31f3c90
//
__int64 __fastcall sub_31F3C90(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *a1 > *a2;
  if ( *a1 < *a2 )
    return 0xFFFFFFFFLL;
  return result;
}
