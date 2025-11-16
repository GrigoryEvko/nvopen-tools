// Function: sub_24C4D90
// Address: 0x24c4d90
//
unsigned __int64 __fastcall sub_24C4D90(__int64 a1, const char *a2, __int64 *a3, int *a4)
{
  int v6; // edx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // r15
  __int64 v10; // rax
  size_t v11; // rax
  __int64 v12; // rdx
  int v13; // eax

  *(_QWORD *)a1 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_DWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 128;
  *(_QWORD *)(a1 + 112) = 1;
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 124) = 1;
  v9 = sub_C57470();
  v10 = *(unsigned int *)(a1 + 80);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v10 + 1, 8u, v7, v8);
    v10 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v10) = v9;
  *(_WORD *)(a1 + 152) = 0;
  ++*(_DWORD *)(a1 + 80);
  *(_BYTE *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_QWORD *)a1 = &unk_49DC090;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_23;
  *(_QWORD *)(a1 + 184) = sub_984030;
  v11 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v11);
  v12 = *a3;
  *(_QWORD *)(a1 + 48) = a3[1];
  v13 = *a4;
  *(_QWORD *)(a1 + 40) = v12;
  *(_BYTE *)(a1 + 12) = (32 * (v13 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return sub_C53130(a1);
}
