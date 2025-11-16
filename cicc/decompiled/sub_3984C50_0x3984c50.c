// Function: sub_3984C50
// Address: 0x3984c50
//
void __fastcall sub_3984C50(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  void (*v5)(); // rax

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(void (**)())(**(_QWORD **)(v4 + 256) + 104LL);
  if ( v5 == nullsub_580 )
  {
    sub_397C0C0(v4, a2, 0);
  }
  else
  {
    ((void (__fastcall *)(_QWORD, __int64, __int64))v5)(*(_QWORD *)(v4 + 256), a3, 1);
    sub_397C0C0(*(_QWORD *)(a1 + 8), a2, 0);
  }
}
