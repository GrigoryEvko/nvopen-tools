// Function: sub_2F39D10
// Address: 0x2f39d10
//
unsigned __int64 __fastcall sub_2F39D10(__int64 a1, const char *a2, __int64 *a3, const char **a4, _BYTE *a5)
{
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  size_t v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  const char *v16; // r15
  size_t v17; // rax
  size_t v18; // r8
  _QWORD *v19; // rdx
  __int64 v21; // rax
  _QWORD *v22; // rdi
  size_t na; // [rsp+8h] [rbp-68h]
  size_t n; // [rsp+8h] [rbp-68h]
  size_t v25; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v26[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v27[8]; // [rsp+30h] [rbp-40h] BYREF

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
    n = (size_t)v10;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v12 + 1, 8u, v12 + 1, v11);
    v12 = *(unsigned int *)(a1 + 80);
    v10 = (__int64 *)n;
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
  v14 = a3[1];
  v15 = *a3;
  v16 = *a4;
  v26[0] = (unsigned __int64)v27;
  *(_QWORD *)(a1 + 40) = v15;
  *(_QWORD *)(a1 + 48) = v14;
  if ( !v16 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v17 = strlen(v16);
  v25 = v17;
  v18 = v17;
  if ( v17 > 0xF )
  {
    na = v17;
    v21 = sub_22409D0((__int64)v26, &v25, 0);
    v18 = na;
    v26[0] = v21;
    v22 = (_QWORD *)v21;
    v27[0] = v25;
  }
  else
  {
    if ( v17 == 1 )
    {
      LOBYTE(v27[0]) = *v16;
      v19 = v27;
      goto LABEL_7;
    }
    if ( !v17 )
    {
      v19 = v27;
      goto LABEL_7;
    }
    v22 = v27;
  }
  memcpy(v22, v16, v18);
  v17 = v25;
  v19 = (_QWORD *)v26[0];
LABEL_7:
  v26[1] = v17;
  *((_BYTE *)v19 + v17) = 0;
  sub_2240AE0((unsigned __int64 *)(a1 + 136), v26);
  *(_BYTE *)(a1 + 208) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 176), v26);
  if ( (_QWORD *)v26[0] != v27 )
    j_j___libc_free_0(v26[0]);
  *(_BYTE *)(a1 + 12) = (32 * (*a5 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return sub_C53130(a1);
}
