// Function: sub_38DC360
// Address: 0x38dc360
//
__int64 (*__fastcall sub_38DC360(__int64 a1))(void)
{
  __int64 (*result)(void); // rax

  result = *(__int64 (**)(void))(*(_QWORD *)a1 + 400LL);
  if ( (char *)result != (char *)nullsub_1953 )
    return (__int64 (*)(void))result();
  return result;
}
