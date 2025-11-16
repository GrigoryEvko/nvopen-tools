// Function: sub_2C122E0
// Address: 0x2c122e0
//
_QWORD *__fastcall sub_2C122E0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r12
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = sub_22077B0(0xA8u);
  v2 = (_QWORD *)v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 24) = 0;
    v3 = v1 + 64;
    v4 = *(_QWORD *)(a1 + 160);
    *(_QWORD *)(v3 - 32) = 0;
    v5 = *(_QWORD *)(a1 + 152);
    *(_BYTE *)(v3 - 56) = 2;
    *(_QWORD *)(v3 - 48) = 0;
    v2[6] = v3;
    v2[7] = 0x200000000LL;
    v9 = 0;
    v10[0] = 0;
    v2[5] = &unk_4A23AA8;
    *v2 = &unk_4A23A70;
    v2[10] = 0;
    v2[11] = 0;
    sub_9C6650(v10);
    sub_2BF0340((__int64)(v2 + 12), 1, 0, (__int64)v2, v6, v7);
    *v2 = &unk_4A231C8;
    v2[5] = &unk_4A23200;
    v2[12] = &unk_4A23238;
    sub_9C6650(&v9);
    v2[19] = v5;
    v2[20] = v4;
    *v2 = &unk_4A24AB8;
    v2[5] = &unk_4A24AF0;
    v2[12] = &unk_4A24B28;
  }
  return v2;
}
