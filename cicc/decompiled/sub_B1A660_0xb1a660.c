// Function: sub_B1A660
// Address: 0xb1a660
//
__int64 __fastcall sub_B1A660(__int64 a1, const char *a2, _BYTE **a3, _BYTE *a4, __int64 *a5)
{
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  size_t v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  _BYTE *v15; // rax
  __int64 v16; // [rsp+0h] [rbp-70h]
  const char *v18; // [rsp+10h] [rbp-60h] BYREF
  char v19; // [rsp+30h] [rbp-40h]
  char v20; // [rsp+31h] [rbp-3Fh]

  *(_QWORD *)a1 = &unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v7;
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
  v8 = sub_C57470();
  v9 = *(unsigned int *)(a1 + 80);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    v16 = v8;
    sub_C8D5F0(a1 + 72, a1 + 88, v9 + 1, 8);
    v9 = *(unsigned int *)(a1 + 80);
    v8 = v16;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v9) = v8;
  ++*(_DWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_BYTE *)(a1 + 153) = 0;
  *(_QWORD *)a1 = &unk_49D9AD8;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_39;
  *(_QWORD *)(a1 + 184) = sub_AA4180;
  v10 = strlen(a2);
  sub_C53080(a1, a2, v10);
  if ( *(_QWORD *)(a1 + 136) )
  {
    v11 = sub_CEADF0();
    v20 = 1;
    v18 = "cl::location(x) specified more than once!";
    v19 = 3;
    sub_C53280(a1, &v18, 0, 0, v11);
  }
  else
  {
    *(_BYTE *)(a1 + 153) = 1;
    v15 = *a3;
    *(_QWORD *)(a1 + 136) = *a3;
    *(_BYTE *)(a1 + 152) = *v15;
  }
  v12 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v13 = a5[1];
  *(_QWORD *)(a1 + 40) = v12;
  *(_QWORD *)(a1 + 48) = v13;
  return sub_C53130(a1);
}
