// Function: sub_2F39570
// Address: 0x2f39570
//
__int64 (*__fastcall sub_2F39570(__int64 a1))(void)
{
  __int64 (*result)(void); // rax

  result = *(__int64 (**)(void))(*(_QWORD *)a1 + 80LL);
  if ( (char *)result != (char *)nullsub_1620 )
    return (__int64 (*)(void))result();
  return result;
}
