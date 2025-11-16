// Function: sub_2AFCD00
// Address: 0x2afcd00
//
unsigned __int64 __fastcall sub_2AFCD00(__int64 a1, const char *a2, int *a3, __int64 *a4, const char **a5)
{
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  size_t v13; // rax
  int v14; // eax
  const char *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rax
  size_t v18; // rax
  size_t v19; // r15
  _QWORD *v20; // rdx
  _QWORD *v22; // rdi
  __int64 *v23; // [rsp+8h] [rbp-68h]
  size_t v24; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v25[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v26[8]; // [rsp+30h] [rbp-40h] BYREF

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
    v23 = v10;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v12 + 1, 8u, v12 + 1, v11);
    v12 = *(unsigned int *)(a1 + 80);
    v10 = v23;
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
  v13 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v13);
  v14 = *a3;
  v15 = *a5;
  v16 = *a4;
  *(_BYTE *)(a1 + 12) = (32 * (v14 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v17 = a4[1];
  *(_QWORD *)(a1 + 40) = v16;
  *(_QWORD *)(a1 + 48) = v17;
  v25[0] = (unsigned __int64)v26;
  if ( !v15 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v18 = strlen(v15);
  v24 = v18;
  v19 = v18;
  if ( v18 > 0xF )
  {
    v25[0] = sub_22409D0((__int64)v25, &v24, 0);
    v22 = (_QWORD *)v25[0];
    v26[0] = v24;
  }
  else
  {
    if ( v18 == 1 )
    {
      LOBYTE(v26[0]) = *v15;
      v20 = v26;
      goto LABEL_7;
    }
    if ( !v18 )
    {
      v20 = v26;
      goto LABEL_7;
    }
    v22 = v26;
  }
  memcpy(v22, v15, v19);
  v18 = v24;
  v20 = (_QWORD *)v25[0];
LABEL_7:
  v25[1] = v18;
  *((_BYTE *)v20 + v18) = 0;
  sub_2240AE0((unsigned __int64 *)(a1 + 136), v25);
  *(_BYTE *)(a1 + 208) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 176), v25);
  if ( (_QWORD *)v25[0] != v26 )
    j_j___libc_free_0(v25[0]);
  return sub_C53130(a1);
}
