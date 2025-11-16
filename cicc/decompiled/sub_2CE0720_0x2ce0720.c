// Function: sub_2CE0720
// Address: 0x2ce0720
//
unsigned __int64 __fastcall sub_2CE0720(__int64 a1, _BYTE *a2, const char *a3, __int64 *a4, _BYTE **a5)
{
  int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rdx
  size_t v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rax
  __int64 *v16; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = &unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_DWORD *)(a1 + 8) = v8;
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
  v11 = *(unsigned int *)(a1 + 80);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    v16 = v9;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v11 + 1, 8u, v11 + 1, v10);
    v11 = *(unsigned int *)(a1 + 80);
    v9 = v16;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v11) = v9;
  *(_WORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 136) = 0;
  ++*(_DWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_QWORD *)a1 = &unk_49DC090;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_23;
  *(_QWORD *)(a1 + 184) = sub_984030;
  *(_BYTE *)(a1 + 12) = (32 * (*a2 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = strlen(a3);
  sub_C53080(a1, (__int64)a3, v12);
  v13 = *a4;
  *(_QWORD *)(a1 + 48) = a4[1];
  v14 = *a5;
  *(_QWORD *)(a1 + 40) = v13;
  LOBYTE(v13) = *v14;
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 136) = v13;
  *(_BYTE *)(a1 + 152) = *v14;
  return sub_C53130(a1);
}
