// Function: sub_B706B0
// Address: 0xb706b0
//
__int64 __fastcall sub_B706B0(__int64 *a1)
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
        sub_C7D6A0(*(_QWORD *)(result + 32), 24LL * *(unsigned int *)(result + 40), 8);
      return j_j___libc_free_0(v2, 128);
    }
  }
  return result;
}
