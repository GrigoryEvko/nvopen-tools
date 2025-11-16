// Function: sub_1525950
// Address: 0x1525950
//
void __fastcall sub_1525950(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _DWORD *v3; // r12

  *(_QWORD *)a1 = a2;
  v2 = sub_22077B0(96);
  if ( v2 )
  {
    *(_QWORD *)v2 = a2;
    *(_QWORD *)(v2 + 8) = 0;
    *(_DWORD *)(v2 + 16) = 2;
    *(_QWORD *)(v2 + 24) = 0;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 56) = 0;
    *(_QWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 88) = 0;
  }
  *(_QWORD *)(a1 + 8) = v2;
  sub_167FAB0(a1 + 16, 3, 1);
  v3 = *(_DWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_WORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 1;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  sub_1524D80(v3, 0x42u, 8);
  sub_1524D80(v3, 0x43u, 8);
  sub_1524D80(v3, 0, 4);
  sub_1524D80(v3, 0xCu, 4);
  sub_1524D80(v3, 0xEu, 4);
  sub_1524D80(v3, 0xDu, 4);
}
