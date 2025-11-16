// Function: sub_3851340
// Address: 0x3851340
//
__int64 *__fastcall sub_3851340(__int64 a1, const char *a2, _DWORD *a3, int **a4, _BYTE *a5, __int64 *a6)
{
  int v9; // edx
  size_t v10; // rax
  int v11; // eax
  int *v12; // rdx
  int v13; // ecx
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
  *(_QWORD *)(a1 + 168) = &unk_49E74C8;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49EEB70;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = &unk_49EEDF0;
  v10 = strlen(a2);
  sub_16B8280(a1, a2, v10);
  v11 = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = *a4;
  *(_BYTE *)(a1 + 12) = v11;
  v13 = *v12;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 160) = v13;
  *(_DWORD *)(a1 + 176) = *v12;
  v14 = *a6;
  *(_BYTE *)(a1 + 12) = *a5 & 7 | v11 & 0xF8;
  v15 = a6[1];
  *(_QWORD *)(a1 + 40) = v14;
  *(_QWORD *)(a1 + 48) = v15;
  return sub_16B88A0(a1);
}
