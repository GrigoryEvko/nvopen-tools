// Function: sub_3259000
// Address: 0x3259000
//
unsigned __int64 __fastcall sub_3259000(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r14
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rax

  v3 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL);
  v4 = sub_E808D0(a3, 0, v3, 0);
  v5 = sub_E808D0(a2, 0, *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL), 0);
  return sub_E81A00(18, v5, v4, v3, 0);
}
