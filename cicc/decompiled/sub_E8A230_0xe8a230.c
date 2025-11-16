// Function: sub_E8A230
// Address: 0xe8a230
//
unsigned __int64 __fastcall sub_E8A230(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r13
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rax

  v5 = *(_QWORD **)(a1 + 8);
  v7 = sub_E808D0(a2, 0, v5, 0);
  v8 = sub_E808D0(a3, 0, v5, 0);
  return sub_E81A00(18, v7, v8, v5, a4);
}
