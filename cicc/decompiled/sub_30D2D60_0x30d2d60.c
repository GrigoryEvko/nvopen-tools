// Function: sub_30D2D60
// Address: 0x30d2d60
//
__int64 __fastcall sub_30D2D60(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 result; // rax

  v1 = *(int *)(a1 + 716);
  v2 = *(int *)(a1 + 652);
  *(_DWORD *)(a1 + 652) = 0;
  result = v1 + v2;
  if ( result > 0x7FFFFFFF )
    result = 0x7FFFFFFF;
  if ( result < (__int64)0xFFFFFFFF80000000LL )
    result = 0xFFFFFFFF80000000LL;
  *(_DWORD *)(a1 + 716) = result;
  return result;
}
