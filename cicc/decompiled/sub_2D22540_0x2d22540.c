// Function: sub_2D22540
// Address: 0x2d22540
//
unsigned __int64 __fastcall sub_2D22540(__int64 a1, const char *a2, _DWORD *a3, _DWORD *a4, __int64 *a5, _BYTE **a6)
{
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  size_t v13; // rax
  __int64 v14; // rdx
  bool v15; // zf
  __int64 v16; // rax
  __int64 *v17; // rax
  _BYTE *v19; // rax
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
  v14 = *a5;
  v15 = *(_QWORD *)(a1 + 136) == 0;
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(a1 + 12) & 0x98 | *a4 & 7 | (32 * (*a3 & 3));
  v16 = a5[1];
  *(_QWORD *)(a1 + 40) = v14;
  *(_QWORD *)(a1 + 48) = v16;
  if ( v15 )
  {
    *(_BYTE *)(a1 + 153) = 1;
    v19 = *a6;
    *(_QWORD *)(a1 + 136) = *a6;
    *(_BYTE *)(a1 + 152) = *v19;
  }
  else
  {
    v17 = sub_CEADF0();
    v24 = 1;
    v22 = "cl::location(x) specified more than once!";
    v23 = 3;
    sub_C53280(a1, (__int64)&v22, 0, 0, (__int64)v17);
  }
  return sub_C53130(a1);
}
