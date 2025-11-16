// Function: sub_2C12110
// Address: 0x2c12110
//
_QWORD *__fastcall sub_2C12110(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r9
  _QWORD *v5; // r12
  __int64 v6; // r14
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v13; // [rsp+8h] [rbp-38h] BYREF
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v15[5]; // [rsp+18h] [rbp-28h] BYREF

  v1 = *(_QWORD *)(a1 + 136);
  v2 = **(_QWORD **)(a1 + 48);
  v3 = sub_22077B0(0x98u);
  v5 = (_QWORD *)v3;
  if ( v3 )
  {
    *(_QWORD *)(v3 + 24) = 0;
    v6 = v3 + 40;
    *(_QWORD *)(v3 + 32) = 0;
    *(_BYTE *)(v3 + 8) = 32;
    *(_QWORD *)(v3 + 64) = v2;
    *(_QWORD *)v3 = &unk_4A231A8;
    *(_QWORD *)(v3 + 16) = 0;
    v13 = 0;
    *(_QWORD *)(v3 + 40) = &unk_4A23170;
    *(_QWORD *)(v3 + 48) = v3 + 64;
    *(_QWORD *)(v3 + 56) = 0x200000001LL;
    v7 = *(unsigned int *)(v2 + 24);
    v8 = *(unsigned int *)(v2 + 28);
    v14 = 0;
    v15[0] = 0;
    if ( v7 + 1 > v8 )
    {
      sub_C8D5F0(v2 + 16, (const void *)(v2 + 32), v7 + 1, 8u, v7 + 1, v4);
      v7 = *(unsigned int *)(v2 + 24);
    }
    *(_QWORD *)(*(_QWORD *)(v2 + 16) + 8 * v7) = v6;
    ++*(_DWORD *)(v2 + 24);
    v5[10] = 0;
    v5[5] = &unk_4A23AA8;
    v9 = v15[0];
    *v5 = &unk_4A23A70;
    v5[11] = v9;
    if ( v9 )
      sub_2AAAFA0(v5 + 11);
    sub_9C6650(v15);
    sub_2BF0340((__int64)(v5 + 12), 1, v1, (__int64)v5, v10, v11);
    *v5 = &unk_4A231C8;
    v5[5] = &unk_4A23200;
    v5[12] = &unk_4A23238;
    sub_9C6650(&v14);
    *v5 = &unk_4A23FE8;
    v5[5] = &unk_4A24030;
    v5[12] = &unk_4A24068;
    sub_9C6650(&v13);
    *v5 = &unk_4A24BD8;
    v5[5] = &unk_4A24C28;
    v5[12] = &unk_4A24C60;
  }
  return v5;
}
