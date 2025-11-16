// Function: sub_38DDC10
// Address: 0x38ddc10
//
void (*__fastcall sub_38DDC10(__int64 a1, __int64 a2, unsigned int *a3))()
{
  void (*result)(); // rax
  __int64 v5; // rdi

  sub_38DDAF0(a1, a3);
  result = (void (*)())sub_38E2470(a2, a3);
  v5 = *(_QWORD *)(a1 + 16);
  if ( v5 )
  {
    result = *(void (**)())(*(_QWORD *)v5 + 24LL);
    if ( result != nullsub_1939 )
      return (void (*)())((__int64 (__fastcall *)(__int64, __int64, unsigned int *))result)(v5, a2, a3);
  }
  return result;
}
