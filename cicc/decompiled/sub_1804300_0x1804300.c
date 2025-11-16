// Function: sub_1804300
// Address: 0x1804300
//
__int64 *__fastcall sub_1804300(__int64 a1, const char *a2, __int64 *a3, int *a4, _BYTE **a5)
{
  int v8; // edx
  size_t v9; // rax
  __int64 v10; // rdx
  int v11; // eax
  _BYTE *v12; // rax

  *(_QWORD *)a1 = &unk_49EED30;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49E74E8;
  *(_WORD *)(a1 + 176) = 256;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49EEC70;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 184) = &unk_49EEDB0;
  v9 = strlen(a2);
  sub_16B8280(a1, a2, v9);
  v10 = *a3;
  *(_QWORD *)(a1 + 48) = a3[1];
  v11 = *a4;
  *(_QWORD *)(a1 + 40) = v10;
  *(_BYTE *)(a1 + 12) = (32 * (v11 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = *a5;
  LOBYTE(v10) = **a5;
  *(_BYTE *)(a1 + 177) = 1;
  *(_BYTE *)(a1 + 160) = v10;
  *(_BYTE *)(a1 + 176) = *v12;
  return sub_16B88A0(a1);
}
