// Function: sub_1D123B0
// Address: 0x1d123b0
//
__int64 (*__fastcall sub_1D123B0(__int64 a1))(void)
{
  __int64 (*result)(void); // rax

  result = *(__int64 (**)(void))(*(_QWORD *)a1 + 80LL);
  if ( (char *)result != (char *)nullsub_683 )
    return (__int64 (*)(void))result();
  return result;
}
