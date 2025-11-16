// Function: sub_2C12410
// Address: 0x2c12410
//
__int64 __fastcall sub_2C12410(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r9
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = **(_QWORD **)(a1 + 48);
  if ( !v1 )
  {
    v3 = sub_22077B0(0x98u);
    if ( v3 )
    {
      v10 = 0;
      v4 = 0;
      goto LABEL_4;
    }
    return 0;
  }
  v3 = sub_22077B0(0x98u);
  if ( !v3 )
    return 0;
  v10 = 0;
  v4 = v1;
LABEL_4:
  *(_BYTE *)(v3 + 8) = 15;
  *(_QWORD *)(v3 + 24) = 0;
  *(_QWORD *)(v3 + 64) = v4;
  *(_QWORD *)v3 = &unk_4A231A8;
  *(_QWORD *)(v3 + 32) = 0;
  v11[0] = 0;
  *(_QWORD *)(v3 + 40) = &unk_4A23170;
  *(_QWORD *)(v3 + 48) = v3 + 64;
  *(_QWORD *)(v3 + 16) = 0;
  *(_QWORD *)(v3 + 56) = 0x200000001LL;
  v5 = *(unsigned int *)(v1 + 24);
  if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 28) )
  {
    sub_C8D5F0(v1 + 16, (const void *)(v1 + 32), v5 + 1, 8u, v5 + 1, v2);
    v5 = *(unsigned int *)(v1 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(v1 + 16) + 8 * v5) = v3 + 40;
  ++*(_DWORD *)(v1 + 24);
  *(_QWORD *)(v3 + 80) = 0;
  *(_QWORD *)(v3 + 40) = &unk_4A23AA8;
  v6 = v11[0];
  *(_QWORD *)v3 = &unk_4A23A70;
  *(_QWORD *)(v3 + 88) = v6;
  if ( v6 )
    sub_2AAAFA0((__int64 *)(v3 + 88));
  sub_9C6650(v11);
  sub_2BF0340(v3 + 96, 1, 0, v3, v7, v8);
  *(_QWORD *)v3 = &unk_4A231C8;
  *(_QWORD *)(v3 + 40) = &unk_4A23200;
  *(_QWORD *)(v3 + 96) = &unk_4A23238;
  sub_9C6650(&v10);
  *(_QWORD *)v3 = &unk_4A24B48;
  *(_QWORD *)(v3 + 40) = &unk_4A24B80;
  *(_QWORD *)(v3 + 96) = &unk_4A24BB8;
  return v3;
}
