// Function: sub_2EC1820
// Address: 0x2ec1820
//
void (*__fastcall sub_2EC1820(__int64 a1, __int64 a2))()
{
  __int64 v2; // rdi
  void (*result)(); // rax

  sub_2F90C60();
  v2 = *(_QWORD *)(a1 + 3472);
  result = *(void (**)())(*(_QWORD *)v2 + 80LL);
  if ( result != nullsub_1615 )
    return (void (*)())((__int64 (__fastcall *)(__int64, __int64))result)(v2, a2);
  return result;
}
