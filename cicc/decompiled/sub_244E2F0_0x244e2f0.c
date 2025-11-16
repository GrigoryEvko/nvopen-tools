// Function: sub_244E2F0
// Address: 0x244e2f0
//
unsigned __int64 __fastcall sub_244E2F0(__int64 a1, const char *a2, char **a3, __int64 *a4, _BYTE *a5)
{
  int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 v10; // rdx
  size_t v11; // rax
  char *v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 *v16; // [rsp+0h] [rbp-60h]
  __int64 v18[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v19[8]; // [rsp+20h] [rbp-40h] BYREF

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
    v16 = v8;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v10 + 1, 8u, v10 + 1, v9);
    v10 = *(unsigned int *)(a1 + 80);
    v8 = v16;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v10) = v8;
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
  v11 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v11);
  v12 = *a3;
  v18[0] = (__int64)v19;
  v13 = -1;
  if ( v12 )
    v13 = (__int64)&v12[strlen(v12)];
  sub_244E240(v18, v12, v13);
  sub_2240AE0((unsigned __int64 *)(a1 + 136), (unsigned __int64 *)v18);
  *(_BYTE *)(a1 + 208) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 176), (unsigned __int64 *)v18);
  if ( (_QWORD *)v18[0] != v19 )
    j_j___libc_free_0(v18[0]);
  v14 = *a4;
  *(_QWORD *)(a1 + 48) = a4[1];
  *(_QWORD *)(a1 + 40) = v14;
  *(_BYTE *)(a1 + 12) = (32 * (*a5 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return sub_C53130(a1);
}
