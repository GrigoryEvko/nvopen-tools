// Function: sub_38C6B90
// Address: 0x38c6b90
//
void __fastcall sub_38C6B90(_QWORD *a1, int a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v5; // r13
  _QWORD v7[4]; // [rsp+10h] [rbp-170h] BYREF
  int v8; // [rsp+30h] [rbp-150h]
  unsigned __int64 *v9; // [rsp+38h] [rbp-148h]
  unsigned __int64 v10[2]; // [rsp+40h] [rbp-140h] BYREF
  _BYTE v11[304]; // [rsp+50h] [rbp-130h] BYREF

  v5 = a1[1];
  v10[1] = 0x10000000000LL;
  v7[0] = &unk_49EFC48;
  v9 = v10;
  v10[0] = (unsigned __int64)v11;
  v8 = 1;
  memset(&v7[1], 0, 24);
  sub_16E7A40((__int64)v7, 0, 0, 0);
  sub_38C6700(v5, (unsigned __int16)a2 | (BYTE2(a2) << 16), a3, a4, (__int64)v7);
  (*(void (__fastcall **)(_QWORD *, unsigned __int64, _QWORD))(*a1 + 400LL))(a1, *v9, *((unsigned int *)v9 + 2));
  v7[0] = &unk_49EFD28;
  sub_16E7960((__int64)v7);
  if ( (_BYTE *)v10[0] != v11 )
    _libc_free(v10[0]);
}
