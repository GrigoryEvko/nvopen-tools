// Function: sub_E97E20
// Address: 0xe97e20
//
__int64 (*__fastcall sub_E97E20(__int64 a1))(void)
{
  __int64 (*result)(void); // rax

  result = *(__int64 (**)(void))(*(_QWORD *)a1 + 512LL);
  if ( (char *)result != (char *)nullsub_360 )
    return (__int64 (*)(void))result();
  return result;
}
