// Function: sub_BA8740
// Address: 0xba8740
//
_QWORD *__fastcall sub_BA8740(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 *v9; // rdi
  char v10; // al
  _BYTE v12[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v13; // [rsp+20h] [rbp-30h]

  *(_QWORD *)(a1 + 16) = a1 + 8;
  *(_QWORD *)(a1 + 8) = (a1 + 8) | 4;
  *(_QWORD *)(a1 + 32) = a1 + 24;
  *(_QWORD *)(a1 + 24) = (a1 + 24) | 4;
  *(_QWORD *)(a1 + 48) = a1 + 40;
  *(_QWORD *)(a1 + 40) = (a1 + 40) | 4;
  *(_QWORD *)(a1 + 64) = a1 + 56;
  *(_QWORD *)(a1 + 56) = (a1 + 56) | 4;
  *(_QWORD *)(a1 + 80) = a1 + 72;
  *(_QWORD *)a1 = a4;
  *(_QWORD *)(a1 + 72) = (a1 + 72) | 4;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 104) = 0;
  v6 = sub_22077B0(32);
  v7 = v6;
  if ( v6 )
  {
    sub_C926D0(v6, 0, 16);
    *(_QWORD *)(v7 + 24) = 0xFFFFFFFFLL;
  }
  *(_QWORD *)(a1 + 120) = v7;
  v8 = (__int64)&a2[a3];
  *(_QWORD *)(a1 + 144) = 0x4800000000LL;
  *(_QWORD *)(a1 + 168) = a1 + 184;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  sub_BA83A0((__int64 *)(a1 + 168), a2, v8);
  *(_QWORD *)(a1 + 200) = a1 + 216;
  sub_BA83A0((__int64 *)(a1 + 200), a2, v8);
  v13 = 257;
  sub_CC9F70(a1 + 232, v12);
  *(_QWORD *)(a1 + 304) = 0x1000000000LL;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  sub_AE1D50(a1 + 312);
  v9 = *(__int64 **)a1;
  *(_QWORD *)(a1 + 824) = 0x1000000000LL;
  v10 = qword_4F80F48[8];
  *(_QWORD *)(a1 + 808) = 0;
  *(_QWORD *)(a1 + 816) = 0;
  *(_QWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = 0;
  *(_QWORD *)(a1 + 848) = 0;
  *(_DWORD *)(a1 + 856) = 0;
  *(_QWORD *)(a1 + 864) = 0;
  *(_BYTE *)(a1 + 872) = v10;
  return sub_B6E750(v9, a1);
}
