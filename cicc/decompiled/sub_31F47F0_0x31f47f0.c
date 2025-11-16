// Function: sub_31F47F0
// Address: 0x31f47f0
//
void (*__fastcall sub_31F47F0(__int64 a1, __int64 a2))()
{
  __int64 v2; // rdi
  void (*result)(); // rax

  v2 = *(_QWORD *)(a1 + 8);
  result = *(void (**)())(*(_QWORD *)v2 + 120LL);
  if ( result != nullsub_98 )
    return (void (*)())((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v2, a2, 1);
  return result;
}
