// Function: sub_F11640
// Address: 0xf11640
//
unsigned __int64 __fastcall sub_F11640(__int64 a1, const char *a2, char **a3, _DWORD *a4, int *a5, __int64 *a6)
{
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  size_t v13; // rax
  char *v14; // rax
  char v15; // dl
  int v16; // edx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v20; // [rsp+0h] [rbp-40h]

  *(_QWORD *)a1 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_DWORD *)(a1 + 8) = v9;
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
  v10 = sub_C57470();
  v12 = *(unsigned int *)(a1 + 80);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    v20 = v10;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v12 + 1, 8u, v12 + 1, v11);
    v12 = *(unsigned int *)(a1 + 80);
    v10 = v20;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v12) = v10;
  *(_WORD *)(a1 + 152) = 0;
  ++*(_DWORD *)(a1 + 80);
  *(_BYTE *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_QWORD *)a1 = &unk_49DC090;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_23;
  *(_QWORD *)(a1 + 184) = sub_984030;
  v13 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v13);
  v14 = *a3;
  v15 = **a3;
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 136) = v15;
  v16 = *a5;
  *(_BYTE *)(a1 + 152) = *v14;
  LODWORD(v14) = *(_BYTE *)(a1 + 12) & 0x98 | v16 & 7 | (32 * (*a4 & 3));
  v17 = *a6;
  *(_BYTE *)(a1 + 12) = (_BYTE)v14;
  v18 = a6[1];
  *(_QWORD *)(a1 + 40) = v17;
  *(_QWORD *)(a1 + 48) = v18;
  return sub_C53130(a1);
}
