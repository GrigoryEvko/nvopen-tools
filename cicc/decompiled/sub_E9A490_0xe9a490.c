// Function: sub_E9A490
// Address: 0xe9a490
//
void (*__fastcall sub_E9A490(__int64 a1, __int64 a2, unsigned __int8 *a3))()
{
  void (*result)(); // rax
  __int64 v5; // rdi

  sub_E9A370(a1, a3);
  result = (void (*)())sub_EA12A0(a2, a3);
  v5 = *(_QWORD *)(a1 + 16);
  if ( v5 )
  {
    result = *(void (**)())(*(_QWORD *)v5 + 24LL);
    if ( result != nullsub_342 )
      return (void (*)())((__int64 (__fastcall *)(__int64, __int64, unsigned __int8 *))result)(v5, a2, a3);
  }
  return result;
}
