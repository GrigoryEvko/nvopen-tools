// Function: sub_14A2A00
// Address: 0x14a2a00
//
__int64 (*__fastcall sub_14A2A00(__int64 a1))(void)
{
  __int64 (*result)(void); // rax

  result = *(__int64 (**)(void))(**(_QWORD **)a1 + 152LL);
  if ( (char *)result != (char *)nullsub_541 )
    return (__int64 (*)(void))result();
  return result;
}
