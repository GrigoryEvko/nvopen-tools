// Function: sub_1604260
// Address: 0x1604260
//
__int64 __fastcall sub_1604260(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // r12

  result = *a1;
  if ( (*a1 & 4) != 0 )
  {
    result &= 0xFFFFFFFFFFFFFFF8LL;
    v2 = result;
    if ( result )
    {
      if ( (*(_BYTE *)(result + 24) & 1) == 0 )
        j___libc_free_0(*(_QWORD *)(result + 32));
      return j_j___libc_free_0(v2, 128);
    }
  }
  return result;
}
