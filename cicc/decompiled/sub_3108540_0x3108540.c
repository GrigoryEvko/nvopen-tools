// Function: sub_3108540
// Address: 0x3108540
//
unsigned __int64 __fastcall sub_3108540(__int64 a1, const char *a2, _QWORD *a3, _BYTE **a4, _BYTE **a5, _BYTE *a6)
{
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  size_t v13; // rax
  __int64 v14; // rax
  bool v15; // zf
  __int64 *v16; // rax
  _BYTE *v17; // rdx
  _BYTE *v18; // rax
  __int64 *v21; // [rsp+8h] [rbp-68h]
  const char *v22; // [rsp+10h] [rbp-60h] BYREF
  char v23; // [rsp+30h] [rbp-40h]
  char v24; // [rsp+31h] [rbp-3Fh]

  *(_QWORD *)a1 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v9;
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
    v21 = v10;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v12 + 1, 8u, v12 + 1, v11);
    v12 = *(unsigned int *)(a1 + 80);
    v10 = v21;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v12) = v10;
  ++*(_DWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_BYTE *)(a1 + 153) = 0;
  *(_QWORD *)a1 = &unk_49D9AD8;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_39;
  *(_QWORD *)(a1 + 184) = sub_AA4180;
  v13 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v13);
  v14 = a3[1];
  v15 = *(_QWORD *)(a1 + 136) == 0;
  *(_QWORD *)(a1 + 40) = *a3;
  *(_QWORD *)(a1 + 48) = v14;
  if ( v15 )
  {
    *(_BYTE *)(a1 + 153) = 1;
    v17 = *a4;
    *(_QWORD *)(a1 + 136) = *a4;
    *(_BYTE *)(a1 + 152) = *v17;
  }
  else
  {
    v16 = sub_CEADF0();
    v24 = 1;
    v22 = "cl::location(x) specified more than once!";
    v23 = 3;
    sub_C53280(a1, (__int64)&v22, 0, 0, (__int64)v16);
    v17 = *(_BYTE **)(a1 + 136);
  }
  v18 = *a5;
  *v17 = **a5;
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) = *v18;
  *(_BYTE *)(a1 + 12) = (32 * (*a6 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return sub_C53130(a1);
}
