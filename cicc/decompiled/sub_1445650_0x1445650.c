// Function: sub_1445650
// Address: 0x1445650
//
__int64 __fastcall sub_1445650(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  _QWORD *v4; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]

  v2 = sub_1444E60(a2, *a2 & 0xFFFFFFFFFFFFFFF8LL);
  *(_QWORD *)(a1 + 40) = v2;
  *(_QWORD *)(a1 + 8) = a1 + 40;
  *(_QWORD *)(a1 + 16) = a1 + 40;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 24) = 0x100000008LL;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = 1;
  v4 = v2;
  v5 = 0;
  sub_14452F0((__int64 *)(a1 + 104), (__int64)&v4);
  return a1;
}
