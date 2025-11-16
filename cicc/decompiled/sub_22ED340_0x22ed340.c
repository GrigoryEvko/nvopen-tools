// Function: sub_22ED340
// Address: 0x22ed340
//
unsigned __int64 __fastcall sub_22ED340(__int64 a1, const char *a2, char **a3)
{
  int v4; // edx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 *v7; // r14
  __int64 v8; // rax
  size_t v9; // rax
  char *v10; // rax
  char v11; // dl

  *(_QWORD *)a1 = &unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_DWORD *)(a1 + 8) = v4;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 128;
  *(_QWORD *)(a1 + 112) = 1;
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 124) = 1;
  v7 = sub_C57470();
  v8 = *(unsigned int *)(a1 + 80);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v8 + 1, 8u, v5, v6);
    v8 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v8) = v7;
  *(_WORD *)(a1 + 152) = 0;
  ++*(_DWORD *)(a1 + 80);
  *(_BYTE *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_QWORD *)a1 = &unk_49DC090;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_23;
  *(_QWORD *)(a1 + 184) = sub_984030;
  v9 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v9);
  v10 = *a3;
  v11 = **a3;
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 136) = v11;
  *(_BYTE *)(a1 + 152) = *v10;
  return sub_C53130(a1);
}
