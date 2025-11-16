// Function: sub_3085AC0
// Address: 0x3085ac0
//
unsigned __int64 __fastcall sub_3085AC0(__int64 a1, const char *a2, _DWORD *a3, __int64 *a4, _BYTE **a5)
{
  int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 v10; // rdx
  size_t v11; // rax
  __int64 v12; // rdx
  bool v13; // zf
  __int64 v14; // rax
  __int64 *v15; // rax
  _BYTE *v17; // rax
  __int64 *v18; // [rsp+0h] [rbp-70h]
  const char *v20; // [rsp+10h] [rbp-60h] BYREF
  char v21; // [rsp+30h] [rbp-40h]
  char v22; // [rsp+31h] [rbp-3Fh]

  *(_QWORD *)a1 = &unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
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
  v10 = *(unsigned int *)(a1 + 80);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    v18 = v8;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v10 + 1, 8u, v10 + 1, v9);
    v10 = *(unsigned int *)(a1 + 80);
    v8 = v18;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v10) = v8;
  ++*(_DWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_BYTE *)(a1 + 153) = 0;
  *(_QWORD *)a1 = &unk_49D9AD8;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_39;
  *(_QWORD *)(a1 + 184) = sub_AA4180;
  v11 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v11);
  v12 = *a4;
  v13 = *(_QWORD *)(a1 + 136) == 0;
  *(_BYTE *)(a1 + 12) = *a3 & 7 | *(_BYTE *)(a1 + 12) & 0xF8;
  v14 = a4[1];
  *(_QWORD *)(a1 + 40) = v12;
  *(_QWORD *)(a1 + 48) = v14;
  if ( v13 )
  {
    *(_BYTE *)(a1 + 153) = 1;
    v17 = *a5;
    *(_QWORD *)(a1 + 136) = *a5;
    *(_BYTE *)(a1 + 152) = *v17;
  }
  else
  {
    v15 = sub_CEADF0();
    v22 = 1;
    v20 = "cl::location(x) specified more than once!";
    v21 = 3;
    sub_C53280(a1, (__int64)&v20, 0, 0, (__int64)v15);
  }
  return sub_C53130(a1);
}
