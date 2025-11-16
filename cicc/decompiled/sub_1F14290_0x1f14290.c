// Function: sub_1F14290
// Address: 0x1f14290
//
__int64 __fastcall sub_1F14290(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _QWORD *v9; // rax
  __int64 v10; // rdi
  __int64 (*v11)(); // rcx
  __int64 v12; // rdx

  v9 = *(_QWORD **)(a2 + 256);
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)a1 = v9;
  *(_QWORD *)(a1 + 24) = a4;
  v10 = v9[2];
  v11 = *(__int64 (**)())(*(_QWORD *)v10 + 40LL);
  v12 = 0;
  if ( v11 != sub_1D00B00 )
  {
    v12 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v11)(v10, a3, 0);
    v9 = *(_QWORD **)a1;
  }
  *(_QWORD *)(a1 + 32) = v12;
  *(_QWORD *)(a1 + 40) = 0;
  sub_1F139C0((_QWORD *)(a1 + 48), a3, (__int64)(v9[13] - v9[12]) >> 3, (__int64)v11, a5, a6);
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0x800000000LL;
  *(_QWORD *)(a1 + 280) = a1 + 296;
  *(_QWORD *)(a1 + 288) = 0x800000000LL;
  *(_QWORD *)(a1 + 632) = 0;
  *(_DWORD *)(a1 + 640) = 0;
  return 0x800000000LL;
}
