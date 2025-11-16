// Function: sub_1E6C100
// Address: 0x1e6c100
//
void (*__fastcall sub_1E6C100(__int64 a1, __int64 a2))()
{
  __int64 v2; // rdi
  void (*result)(); // rax

  sub_1F03410(a1, a2);
  v2 = *(_QWORD *)(a1 + 2120);
  result = *(void (**)())(*(_QWORD *)v2 + 72LL);
  if ( result != nullsub_711 )
    return (void (*)())((__int64 (__fastcall *)(__int64, __int64))result)(v2, a2);
  return result;
}
