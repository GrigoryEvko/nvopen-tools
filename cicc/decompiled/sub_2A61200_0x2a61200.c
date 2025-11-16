// Function: sub_2A61200
// Address: 0x2a61200
//
_BYTE *__fastcall sub_2A61200(__int64 **a1)
{
  _BYTE *result; // rax
  __int64 *v2; // r14
  _QWORD *v3; // r13
  __int64 v4; // r14
  _QWORD *v5; // rbx
  _QWORD *v6; // [rsp+0h] [rbp-60h] BYREF
  __int64 v7; // [rsp+8h] [rbp-58h]
  const char *v8; // [rsp+10h] [rbp-50h] BYREF
  char v9; // [rsp+30h] [rbp-30h]
  char v10; // [rsp+31h] [rbp-2Fh]

  result = sub_BA8CD0((__int64)a1, (__int64)"__llvm_fs_discriminator__", 0x19u, 0);
  if ( !result )
  {
    v2 = *a1;
    v3 = (_QWORD *)sub_BCB2A0(*a1);
    v8 = "__llvm_fs_discriminator__";
    v4 = sub_ACD6D0(v2);
    v10 = 1;
    v9 = 3;
    BYTE4(v7) = 0;
    v5 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v5 )
      sub_B30000((__int64)v5, (__int64)a1, v3, 1, 5, v4, (__int64)&v8, 0, 0, v7, 0);
    v6 = v5;
    return (_BYTE *)sub_2A413E0(a1, (unsigned __int64 *)&v6, 1);
  }
  return result;
}
