// Function: sub_1706210
// Address: 0x1706210
//
__int64 *__fastcall sub_1706210(__int64 a1, const char *a2, char **a3, _DWORD *a4, int *a5, __int64 *a6)
{
  int v9; // edx
  size_t v10; // rax
  char *v11; // rax
  char v12; // dl
  int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // rax

  *(_QWORD *)a1 = &unk_49EED30;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v9;
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
  v10 = strlen(a2);
  sub_16B8280(a1, a2, v10);
  v11 = *a3;
  v12 = **a3;
  *(_BYTE *)(a1 + 177) = 1;
  *(_BYTE *)(a1 + 160) = v12;
  v13 = *a5;
  *(_BYTE *)(a1 + 176) = *v11;
  LODWORD(v11) = *(_BYTE *)(a1 + 12) & 0x98 | v13 & 7 | (32 * (*a4 & 3));
  v14 = *a6;
  *(_BYTE *)(a1 + 12) = (_BYTE)v11;
  v15 = a6[1];
  *(_QWORD *)(a1 + 40) = v14;
  *(_QWORD *)(a1 + 48) = v15;
  return sub_16B88A0(a1);
}
