// Function: sub_30D2CE0
// Address: 0x30d2ce0
//
__int64 __fastcall sub_30D2CE0(__int64 a1)
{
  __int64 result; // rax

  result = *(int *)(a1 + 716) + (__int64)(int)qword_5030168;
  if ( result > 0x7FFFFFFF )
    result = 0x7FFFFFFF;
  if ( result < (__int64)0xFFFFFFFF80000000LL )
    result = 0xFFFFFFFF80000000LL;
  *(_DWORD *)(a1 + 716) = result;
  return result;
}
