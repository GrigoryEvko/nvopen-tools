// Function: sub_30D1CC0
// Address: 0x30d1cc0
//
__int64 __fastcall sub_30D1CC0(__int64 a1)
{
  __int64 result; // rax

  result = *(int *)(a1 + 716) - 5000LL;
  if ( result < (__int64)0xFFFFFFFF80000000LL )
    result = 0xFFFFFFFF80000000LL;
  *(_DWORD *)(a1 + 716) = result;
  return result;
}
