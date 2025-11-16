// Function: sub_2C0E520
// Address: 0x2c0e520
//
__int64 __fastcall sub_2C0E520(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 v3; // r8
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v9; // [rsp+18h] [rbp-48h] BYREF
  __int64 v10; // [rsp+20h] [rbp-40h] BYREF
  __int64 v11[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = **(_QWORD **)(a1 + 48);
  v9 = *(_QWORD *)(a1 + 88);
  if ( v9 )
    sub_2AAAFA0(&v9);
  v2 = sub_22077B0(0x98u);
  if ( v2 )
  {
    v10 = v9;
    if ( v9 )
    {
      sub_2AAAFA0(&v10);
      v11[0] = v10;
      if ( v10 )
        sub_2AAAFA0(v11);
    }
    else
    {
      v11[0] = 0;
    }
    *(_BYTE *)(v2 + 8) = 28;
    v3 = v2 + 40;
    *(_QWORD *)(v2 + 24) = 0;
    *(_QWORD *)(v2 + 64) = v1;
    *(_QWORD *)v2 = &unk_4A231A8;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_QWORD *)(v2 + 40) = &unk_4A23170;
    *(_QWORD *)(v2 + 48) = v2 + 64;
    *(_QWORD *)(v2 + 56) = 0x200000001LL;
    v4 = *(unsigned int *)(v1 + 24);
    if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 28) )
    {
      sub_C8D5F0(v1 + 16, (const void *)(v1 + 32), v4 + 1, 8u, v3, v4 + 1);
      v4 = *(unsigned int *)(v1 + 24);
      v3 = v2 + 40;
    }
    *(_QWORD *)(*(_QWORD *)(v1 + 16) + 8 * v4) = v3;
    ++*(_DWORD *)(v1 + 24);
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 40) = &unk_4A23AA8;
    v5 = v11[0];
    *(_QWORD *)v2 = &unk_4A23A70;
    *(_QWORD *)(v2 + 88) = v5;
    if ( v5 )
      sub_2AAAFA0((__int64 *)(v2 + 88));
    sub_9C6650(v11);
    sub_2BF0340(v2 + 96, 1, 0, v2, v6, v7);
    *(_QWORD *)v2 = &unk_4A231C8;
    *(_QWORD *)(v2 + 40) = &unk_4A23200;
    *(_QWORD *)(v2 + 96) = &unk_4A23238;
    sub_9C6650(&v10);
    *(_QWORD *)v2 = &unk_4A24670;
    *(_QWORD *)(v2 + 40) = &unk_4A246B0;
    *(_QWORD *)(v2 + 96) = &unk_4A246E8;
  }
  sub_9C6650(&v9);
  return v2;
}
