// Function: sub_38C6E60
// Address: 0x38c6e60
//
void __fastcall sub_38C6E60(_QWORD *a1, unsigned __int64 a2)
{
  __int64 v2; // r15
  _QWORD v3[4]; // [rsp+0h] [rbp-170h] BYREF
  int v4; // [rsp+20h] [rbp-150h]
  unsigned __int64 *v5; // [rsp+28h] [rbp-148h]
  unsigned __int64 v6[2]; // [rsp+30h] [rbp-140h] BYREF
  _BYTE v7[304]; // [rsp+40h] [rbp-130h] BYREF

  v2 = a1[1];
  v6[1] = 0x10000000000LL;
  v5 = v6;
  v6[0] = (unsigned __int64)v7;
  v3[0] = &unk_49EFC48;
  v4 = 1;
  memset(&v3[1], 0, 24);
  sub_16E7A40((__int64)v3, 0, 0, 0);
  sub_38C6CD0(v2, a2, (__int64)v3);
  (*(void (__fastcall **)(_QWORD *, unsigned __int64, _QWORD))(*a1 + 400LL))(a1, *v5, *((unsigned int *)v5 + 2));
  v3[0] = &unk_49EFD28;
  sub_16E7960((__int64)v3);
  if ( (_BYTE *)v6[0] != v7 )
    _libc_free(v6[0]);
}
