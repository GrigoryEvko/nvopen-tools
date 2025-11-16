// Function: sub_16BC600
// Address: 0x16bc600
//
__int64 __fastcall sub_16BC600(_QWORD *a1)
{
  __int64 result; // rax

  if ( a1 )
  {
    *a1 = off_49EF220;
    nullsub_804();
    return j_j___libc_free_0(a1, 8);
  }
  return result;
}
