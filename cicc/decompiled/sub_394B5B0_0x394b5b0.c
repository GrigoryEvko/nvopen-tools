// Function: sub_394B5B0
// Address: 0x394b5b0
//
__int64 *__fastcall sub_394B5B0(__int64 a1, const char *a2, const char **a3, _BYTE *a4, __int64 *a5)
{
  int v8; // edx
  size_t v9; // rax
  const char *v10; // r13
  size_t v11; // rax
  size_t v12; // r9
  _QWORD *v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v17; // rax
  _QWORD *v18; // rdi
  size_t n; // [rsp+0h] [rbp-70h]
  size_t v21; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v22[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v23[8]; // [rsp+30h] [rbp-40h] BYREF

  *(_QWORD *)a1 = &unk_49EED30;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_QWORD *)(a1 + 160) = a1 + 176;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_DWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 192) = &unk_49EED10;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49EEBF0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 240) = &unk_49EEE90;
  *(_QWORD *)(a1 + 248) = a1 + 264;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_BYTE *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_BYTE *)(a1 + 216) = 0;
  *(_BYTE *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_BYTE *)(a1 + 264) = 0;
  v9 = strlen(a2);
  sub_16B8280(a1, a2, v9);
  v10 = *a3;
  v22[0] = (unsigned __int64)v23;
  if ( !v10 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v11 = strlen(v10);
  v21 = v11;
  v12 = v11;
  if ( v11 > 0xF )
  {
    n = v11;
    v17 = sub_22409D0((__int64)v22, &v21, 0);
    v12 = n;
    v22[0] = v17;
    v18 = (_QWORD *)v17;
    v23[0] = v21;
  }
  else
  {
    if ( v11 == 1 )
    {
      LOBYTE(v23[0]) = *v10;
      v13 = v23;
      goto LABEL_5;
    }
    if ( !v11 )
    {
      v13 = v23;
      goto LABEL_5;
    }
    v18 = v23;
  }
  memcpy(v18, v10, v12);
  v11 = v21;
  v13 = (_QWORD *)v22[0];
LABEL_5:
  v22[1] = v11;
  *((_BYTE *)v13 + v11) = 0;
  sub_2240AE0((unsigned __int64 *)(a1 + 160), v22);
  *(_BYTE *)(a1 + 232) = 1;
  sub_2240AE0((unsigned __int64 *)(a1 + 200), v22);
  if ( (_QWORD *)v22[0] != v23 )
    j_j___libc_free_0(v22[0]);
  v14 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v15 = a5[1];
  *(_QWORD *)(a1 + 40) = v14;
  *(_QWORD *)(a1 + 48) = v15;
  return sub_16B88A0(a1);
}
