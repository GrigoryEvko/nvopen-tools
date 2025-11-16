// Function: sub_284F530
// Address: 0x284f530
//
bool __fastcall sub_284F530(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned int v3; // r14d
  _QWORD *v4; // rax
  __int64 v5; // rax

  v2 = sub_D95540(**(_QWORD **)(a1 + 32));
  v3 = sub_D97050((__int64)a2, v2) * *(_DWORD *)(a1 + 40);
  v4 = (_QWORD *)sub_B2BE50(*a2);
  v5 = sub_BCCE00(v4, v3);
  return *((_WORD *)sub_DC5000((__int64)a2, a1, v5, 0) + 12) == 6;
}
