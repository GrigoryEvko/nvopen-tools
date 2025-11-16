// Function: sub_30D2D20
// Address: 0x30d2d20
//
__int64 __fastcall sub_30D2D20(__int64 a1)
{
  __int64 result; // rax

  result = *(int *)(a1 + 716) + (__int64)dword_5030088;
  if ( result > 0x7FFFFFFF )
    result = 0x7FFFFFFF;
  if ( result < (__int64)0xFFFFFFFF80000000LL )
    result = 0xFFFFFFFF80000000LL;
  *(_DWORD *)(a1 + 716) = result;
  return result;
}
