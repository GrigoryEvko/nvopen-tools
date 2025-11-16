// Function: sub_124BD80
// Address: 0x124bd80
//
__int64 __fastcall sub_124BD80(__int64 a1, char a2)
{
  _QWORD *v2; // r13
  __int64 v3; // rax
  __int64 v4; // r12

  v2 = *(_QWORD **)(a1 + 8);
  v3 = (*(__int64 (__fastcall **)(_QWORD *))(*v2 + 80LL))(v2) + v2[4] - v2[2];
  v4 = -(1LL << a2) & ((1LL << a2) + v3 - 1);
  sub_CB6C70(*(_QWORD *)(a1 + 8), v4 - v3);
  return v4;
}
