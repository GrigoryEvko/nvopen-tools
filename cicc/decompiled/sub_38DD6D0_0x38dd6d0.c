// Function: sub_38DD6D0
// Address: 0x38dd6d0
//
void __fastcall sub_38DD6D0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdi
  unsigned __int64 v3; // rdx
  _QWORD v4[2]; // [rsp+0h] [rbp-110h] BYREF
  _QWORD *v5; // [rsp+10h] [rbp-100h] BYREF
  __int16 v6; // [rsp+20h] [rbp-F0h]
  _QWORD v7[4]; // [rsp+30h] [rbp-E0h] BYREF
  int v8; // [rsp+50h] [rbp-C0h]
  unsigned __int64 *v9; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v10[2]; // [rsp+60h] [rbp-B0h] BYREF
  _BYTE v11[160]; // [rsp+70h] [rbp-A0h] BYREF

  v10[1] = 0x8000000000LL;
  v9 = v10;
  v10[0] = (unsigned __int64)v11;
  v7[0] = &unk_49EFC48;
  v8 = 1;
  memset(&v7[1], 0, 24);
  sub_16E7A40((__int64)v7, 0, 0, 0);
  sub_38CDBE0(a2, (__int64)v7, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 16LL));
  v2 = *(__int64 **)(a1 + 8);
  v3 = *v9;
  v4[1] = *((unsigned int *)v9 + 2);
  v6 = 261;
  v4[0] = v3;
  v5 = v4;
  sub_38DD5A0(v2, (__int64)&v5);
  v7[0] = &unk_49EFD28;
  sub_16E7960((__int64)v7);
  if ( (_BYTE *)v10[0] != v11 )
    _libc_free(v10[0]);
}
