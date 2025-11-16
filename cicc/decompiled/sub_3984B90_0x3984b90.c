// Function: sub_3984B90
// Address: 0x3984b90
//
__int64 __fastcall sub_3984B90(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  __int64 v4; // rdi
  void (*v5)(); // rax

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(void (**)())(**(_QWORD **)(v4 + 256) + 104LL);
  if ( v5 == nullsub_580 )
    return sub_396F300(v4, a2);
  ((void (__fastcall *)(_QWORD, __int64, __int64))v5)(*(_QWORD *)(v4 + 256), a3, 1);
  return sub_396F300(*(_QWORD *)(a1 + 8), a2);
}
