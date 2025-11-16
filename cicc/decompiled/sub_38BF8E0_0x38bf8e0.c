// Function: sub_38BF8E0
// Address: 0x38bf8e0
//
__int64 __fastcall sub_38BF8E0(__int64 a1, __int64 a2, char a3, char a4)
{
  __int64 v6; // rax
  char *v7; // rsi
  size_t v8; // rdx
  _QWORD *v9; // r9
  int v10; // r9d
  __int64 v11; // r12
  __int64 v13; // [rsp+10h] [rbp-100h]
  _QWORD v15[2]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v16; // [rsp+30h] [rbp-E0h]
  void *dest; // [rsp+38h] [rbp-D8h]
  int v18; // [rsp+40h] [rbp-D0h]
  void **v19; // [rsp+48h] [rbp-C8h]
  _BYTE *v20; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+58h] [rbp-B8h]
  _BYTE v22[176]; // [rsp+60h] [rbp-B0h] BYREF

  v21 = 0x8000000000LL;
  v19 = (void **)&v20;
  v15[0] = &unk_49EFC48;
  v20 = v22;
  v18 = 1;
  dest = 0;
  v16 = 0;
  v15[1] = 0;
  sub_16E7A40((__int64)v15, 0, 0, 0);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(char **)(v6 + 80);
  v8 = *(_QWORD *)(v6 + 88);
  if ( v8 > v16 - (__int64)dest )
  {
    v9 = (_QWORD *)sub_16E7EE0((__int64)v15, v7, v8);
  }
  else
  {
    v9 = v15;
    if ( v8 )
    {
      v13 = *(_QWORD *)(v6 + 88);
      memcpy(dest, v7, v8);
      v9 = v15;
      dest = (char *)dest + v13;
    }
  }
  sub_16E2CE0(a2, (__int64)v9);
  v15[0] = &unk_49EFD28;
  sub_16E7960((__int64)v15);
  v11 = sub_38BEE30(a1, v20, (unsigned int)v21, a3, a4, v10);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  return v11;
}
