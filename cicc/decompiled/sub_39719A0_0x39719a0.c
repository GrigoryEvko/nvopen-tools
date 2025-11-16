// Function: sub_39719A0
// Address: 0x39719a0
//
void (*__fastcall sub_39719A0(__int64 *a1, _QWORD *a2))()
{
  __int64 v2; // rdi
  void (*result)(); // rax
  bool v4[19]; // [rsp+Dh] [rbp-13h] BYREF

  sub_3970C70(a1, a2, v4);
  v2 = a1[32];
  result = *(void (**)())(*(_QWORD *)v2 + 544LL);
  if ( result != nullsub_588 )
    return (void (*)())((__int64 (__fastcall *)(__int64, bool *))result)(v2, v4);
  return result;
}
