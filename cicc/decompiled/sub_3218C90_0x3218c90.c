// Function: sub_3218C90
// Address: 0x3218c90
//
__int64 __fastcall sub_3218C90(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  __int64 v4; // rdi
  void (*v5)(); // rax

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(void (**)())(**(_QWORD **)(v4 + 224) + 120LL);
  if ( v5 == nullsub_98 )
    return sub_31DC9D0(v4, a2);
  ((void (__fastcall *)(_QWORD, __int64, __int64))v5)(*(_QWORD *)(v4 + 224), a3, 1);
  return sub_31DC9D0(*(_QWORD *)(a1 + 8), a2);
}
