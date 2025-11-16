// Function: sub_1DC3420
// Address: 0x1dc3420
//
__int64 __fastcall sub_1DC3420(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(*a1 + 3) & 0x10) != 0 )
    return sub_1DC3350(*(_QWORD *)(a1[1] + 16), *(__int64 **)(a1[1] + 32), a2, *a1);
  return result;
}
