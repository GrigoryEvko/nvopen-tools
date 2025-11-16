// Function: sub_16F10B0
// Address: 0x16f10b0
//
__int64 __fastcall sub_16F10B0(_QWORD *a1)
{
  __int64 result; // rax

  if ( a1 )
  {
    sub_16F1080(a1);
    return j_j___libc_free_0(a1, 32);
  }
  return result;
}
