// Function: sub_2FF3600
// Address: 0x2ff3600
//
unsigned __int64 __fastcall sub_2FF3600(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4, char **a5, _BYTE *a6)
{
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  char *v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 *v20; // [rsp+0h] [rbp-60h]
  __int64 v22[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v23[8]; // [rsp+20h] [rbp-40h] BYREF

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
    v20 = v10;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v12 + 1, 8u, v12 + 1, v11);
    v12 = *(unsigned int *)(a1 + 80);
    v10 = v20;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v12) = v10;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  ++*(_DWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49DC130;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)a1 = &unk_49DC010;
  *(_BYTE *)(a1 + 192) = 0;
  *(_BYTE *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = &unk_49DC350;
  *(_QWORD *)(a1 + 248) = nullsub_92;
  *(_QWORD *)(a1 + 240) = sub_BC4D70;
  sub_C53080(a1, *a2, a2[1]);
  v13 = a3[1];
  v14 = *a3;
  v15 = *a5;
  *(_QWORD *)(a1 + 40) = v14;
  v16 = *a4;
  *(_QWORD *)(a1 + 48) = v13;
  v17 = a4[1];
  *(_QWORD *)(a1 + 56) = v16;
  v18 = -1;
  *(_QWORD *)(a1 + 64) = v17;
  v22[0] = (__int64)v23;
  if ( v15 )
    v18 = (__int64)&v15[strlen(v15)];
  sub_2FEE630(v22, v15, v18);
  sub_2240AE0((unsigned __int64 *)(a1 + 136), (unsigned __int64 *)v22);
  *(_BYTE *)(a1 + 208) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 176), (unsigned __int64 *)v22);
  if ( (_QWORD *)v22[0] != v23 )
    j_j___libc_free_0(v22[0]);
  *(_BYTE *)(a1 + 12) = (32 * (*a6 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return sub_C53130(a1);
}
