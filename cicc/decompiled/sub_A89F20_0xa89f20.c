// Function: sub_A89F20
// Address: 0xa89f20
//
__int64 __fastcall sub_A89F20(__int64 a1, const char *a2, _QWORD *a3, __int64 a4)
{
  int v5; // edx
  __int64 v6; // r14
  __int64 v7; // rax
  size_t v8; // rax
  __int64 v9; // rax

  *(_QWORD *)a1 = &unk_49DC150;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_DWORD *)(a1 + 8) = v5;
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
  v6 = sub_C57470();
  v7 = *(unsigned int *)(a1 + 80);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, a1 + 88, v7 + 1, 8);
    v7 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v7) = v6;
  *(_WORD *)(a1 + 152) = 0;
  ++*(_DWORD *)(a1 + 80);
  *(_BYTE *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_QWORD *)a1 = &unk_49DC090;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_23;
  *(_QWORD *)(a1 + 184) = sub_984030;
  v8 = strlen(a2);
  sub_C53080(a1, a2, v8);
  v9 = a3[1];
  *(_QWORD *)(a1 + 40) = *a3;
  *(_QWORD *)(a1 + 48) = v9;
  return sub_C53130(a1);
}
