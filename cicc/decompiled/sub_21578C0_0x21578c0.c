// Function: sub_21578C0
// Address: 0x21578c0
//
void __fastcall sub_21578C0(__int64 a1)
{
  __int64 v1; // rdi
  unsigned __int64 v2; // rdx
  _QWORD v3[2]; // [rsp+0h] [rbp-110h] BYREF
  _QWORD *v4; // [rsp+10h] [rbp-100h] BYREF
  __int16 v5; // [rsp+20h] [rbp-F0h]
  _QWORD v6[4]; // [rsp+30h] [rbp-E0h] BYREF
  int v7; // [rsp+50h] [rbp-C0h]
  unsigned __int64 *v8; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v9[2]; // [rsp+60h] [rbp-B0h] BYREF
  _BYTE v10[160]; // [rsp+70h] [rbp-A0h] BYREF

  v9[1] = 0x8000000000LL;
  v8 = v9;
  v9[0] = (unsigned __int64)v10;
  v6[0] = &unk_49EFC48;
  v7 = 1;
  memset(&v6[1], 0, 24);
  sub_16E7A40((__int64)v6, 0, 0, 0);
  sub_2157750(a1, **(_QWORD **)(a1 + 264), (__int64)v6);
  v1 = *(_QWORD *)(a1 + 256);
  v2 = *v8;
  v3[1] = *((unsigned int *)v8 + 2);
  v5 = 261;
  v3[0] = v2;
  v4 = v3;
  sub_38DD5A0(v1, &v4);
  v6[0] = &unk_49EFD28;
  sub_16E7960((__int64)v6);
  if ( (_BYTE *)v9[0] != v10 )
    _libc_free(v9[0]);
}
