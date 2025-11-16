// Function: sub_1F487E0
// Address: 0x1f487e0
//
__int64 *__fastcall sub_1F487E0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, char **a5, _BYTE *a6)
{
  int v10; // edx
  size_t v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  char *v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v19[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v20[8]; // [rsp+10h] [rbp-40h] BYREF

  *(_QWORD *)a1 = &unk_49EED30;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_QWORD *)(a1 + 160) = a1 + 176;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 192) = &unk_49EED10;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49EEBF0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 240) = &unk_49EEE90;
  *(_QWORD *)(a1 + 248) = a1 + 264;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_BYTE *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_BYTE *)(a1 + 216) = 0;
  *(_BYTE *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  v11 = *(_QWORD *)(a2 + 8);
  *(_BYTE *)(a1 + 264) = 0;
  sub_16B8280(a1, *(const void **)a2, v11);
  v12 = a3[1];
  v13 = *a3;
  v14 = *a5;
  *(_QWORD *)(a1 + 40) = v13;
  v15 = *a4;
  *(_QWORD *)(a1 + 48) = v12;
  v16 = a4[1];
  *(_QWORD *)(a1 + 56) = v15;
  v17 = -1;
  *(_QWORD *)(a1 + 64) = v16;
  v19[0] = (__int64)v20;
  if ( v14 )
    v17 = (__int64)&v14[strlen(v14)];
  sub_1F450A0(v19, v14, v17);
  sub_2240AE0(a1 + 160, v19);
  *(_BYTE *)(a1 + 232) = 1;
  sub_2240AE0(a1 + 200, v19);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0], v20[0] + 1LL);
  *(_BYTE *)(a1 + 12) = (32 * (*a6 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return sub_16B88A0(a1);
}
