// Function: sub_25F1470
// Address: 0x25f1470
//
unsigned __int64 __fastcall sub_25F1470(__int64 a1, const char *a2, const char **a3, _BYTE *a4, __int64 *a5)
{
  int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rdx
  size_t v12; // rax
  const char *v13; // r13
  size_t v14; // rax
  size_t v15; // r9
  _QWORD *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v20; // rax
  _QWORD *v21; // rdi
  size_t na; // [rsp+0h] [rbp-70h]
  size_t n; // [rsp+0h] [rbp-70h]
  size_t v25; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v26[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v27[8]; // [rsp+30h] [rbp-40h] BYREF

  *(_QWORD *)a1 = &unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v8;
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
  v9 = sub_C57470();
  v11 = *(unsigned int *)(a1 + 80);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    n = (size_t)v9;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v11 + 1, 8u, v11 + 1, v10);
    v11 = *(unsigned int *)(a1 + 80);
    v9 = (__int64 *)n;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v11) = v9;
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
  v12 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v12);
  v13 = *a3;
  v26[0] = (unsigned __int64)v27;
  if ( !v13 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v14 = strlen(v13);
  v25 = v14;
  v15 = v14;
  if ( v14 > 0xF )
  {
    na = v14;
    v20 = sub_22409D0((__int64)v26, &v25, 0);
    v15 = na;
    v26[0] = v20;
    v21 = (_QWORD *)v20;
    v27[0] = v25;
  }
  else
  {
    if ( v14 == 1 )
    {
      LOBYTE(v27[0]) = *v13;
      v16 = v27;
      goto LABEL_7;
    }
    if ( !v14 )
    {
      v16 = v27;
      goto LABEL_7;
    }
    v21 = v27;
  }
  memcpy(v21, v13, v15);
  v14 = v25;
  v16 = (_QWORD *)v26[0];
LABEL_7:
  v26[1] = v14;
  *((_BYTE *)v16 + v14) = 0;
  sub_2240AE0((unsigned __int64 *)(a1 + 136), v26);
  *(_BYTE *)(a1 + 208) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 176), v26);
  if ( (_QWORD *)v26[0] != v27 )
    j_j___libc_free_0(v26[0]);
  v17 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v18 = a5[1];
  *(_QWORD *)(a1 + 40) = v17;
  *(_QWORD *)(a1 + 48) = v18;
  return sub_C53130(a1);
}
