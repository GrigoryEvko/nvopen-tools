// Function: sub_214C040
// Address: 0x214c040
//
__int64 __fastcall sub_214C040(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *v2; // r15
  unsigned int v3; // eax
  __int64 v4; // rdi
  __int64 v5; // r12
  _QWORD v7[2]; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v8; // [rsp+10h] [rbp-E0h]
  __int64 v9; // [rsp+18h] [rbp-D8h]
  int v10; // [rsp+20h] [rbp-D0h]
  unsigned __int64 *v11; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v12[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v13[176]; // [rsp+40h] [rbp-B0h] BYREF

  v12[1] = 0x8000000000LL;
  v12[0] = (unsigned __int64)v13;
  v10 = 1;
  v7[0] = &unk_49EFC48;
  v9 = 0;
  v8 = 0;
  v7[1] = 0;
  v11 = v12;
  sub_16E7A40((__int64)v7, 0, 0, 0);
  v1 = v9;
  if ( (unsigned __int64)(v8 - v9) <= 0xC )
  {
    v2 = (_QWORD *)sub_16E7EE0((__int64)v7, "__local_depot", 0xDu);
  }
  else
  {
    *(_DWORD *)(v9 + 8) = 1869636964;
    v2 = v7;
    *(_QWORD *)v1 = 0x5F6C61636F6C5F5FLL;
    *(_BYTE *)(v1 + 12) = 116;
    v9 += 13;
  }
  v3 = sub_396DD70(a1);
  sub_16E7A90((__int64)v2, v3);
  v7[0] = &unk_49EFD28;
  sub_16E7960((__int64)v7);
  v4 = *(_QWORD *)(a1 + 248);
  LOWORD(v8) = 262;
  v7[0] = v12;
  v5 = sub_38BF510(v4, v7);
  if ( (_BYTE *)v12[0] != v13 )
    _libc_free(v12[0]);
  return v5;
}
