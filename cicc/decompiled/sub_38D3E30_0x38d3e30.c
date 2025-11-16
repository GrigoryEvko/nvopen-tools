// Function: sub_38D3E30
// Address: 0x38d3e30
//
void (*__fastcall sub_38D3E30(__int64 a1, __int64 a2))()
{
  __int64 v2; // rdi
  void (*result)(); // rax

  sub_390D5F0(*(_QWORD *)(a1 + 264), a2, 0);
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 264) + 24LL);
  result = *(void (**)())(*(_QWORD *)v2 + 64LL);
  if ( result != nullsub_1933 )
    return (void (*)())((__int64 (__fastcall *)(__int64, __int64))result)(v2, a2);
  return result;
}
