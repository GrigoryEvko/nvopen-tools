// Function: sub_2C13FA0
// Address: 0x2c13fa0
//
__int64 __fastcall sub_2C13FA0(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // r14
  __int64 v5; // r10
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v15; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h] BYREF
  __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  __int64 v18[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = **(_QWORD **)(a1 + 48);
  v15 = *(_QWORD *)(a1 + 88);
  if ( v15 )
    sub_2AAAFA0(&v15);
  v4 = sub_22077B0(0x98u);
  if ( v4 )
  {
    v16 = v15;
    if ( v15 )
    {
      sub_2AAAFA0(&v16);
      v17 = v16;
      if ( v16 )
      {
        sub_2AAAFA0(&v17);
        v18[0] = v17;
        if ( v17 )
          sub_2AAAFA0(v18);
        goto LABEL_8;
      }
    }
    else
    {
      v17 = 0;
    }
    v18[0] = 0;
LABEL_8:
    *(_BYTE *)(v4 + 8) = 30;
    v5 = v4 + 40;
    *(_QWORD *)(v4 + 24) = 0;
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)v4 = &unk_4A231A8;
    *(_QWORD *)(v4 + 16) = 0;
    *(_QWORD *)(v4 + 64) = v1;
    *(_QWORD *)(v4 + 40) = &unk_4A23170;
    *(_QWORD *)(v4 + 48) = v4 + 64;
    *(_QWORD *)(v4 + 56) = 0x200000001LL;
    v6 = *(unsigned int *)(v1 + 24);
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 28) )
    {
      sub_C8D5F0(v1 + 16, (const void *)(v1 + 32), v6 + 1, 8u, v2, v3);
      v6 = *(unsigned int *)(v1 + 24);
      v5 = v4 + 40;
    }
    *(_QWORD *)(*(_QWORD *)(v1 + 16) + 8 * v6) = v5;
    ++*(_DWORD *)(v1 + 24);
    *(_QWORD *)(v4 + 80) = 0;
    *(_QWORD *)(v4 + 40) = &unk_4A23AA8;
    v7 = v18[0];
    *(_QWORD *)v4 = &unk_4A23A70;
    *(_QWORD *)(v4 + 88) = v7;
    if ( v7 )
      sub_2AAAFA0((__int64 *)(v4 + 88));
    sub_9C6650(v18);
    sub_2BF0340(v4 + 96, 1, 0, v4, v8, v9);
    *(_QWORD *)v4 = &unk_4A231C8;
    *(_QWORD *)(v4 + 40) = &unk_4A23200;
    *(_QWORD *)(v4 + 96) = &unk_4A23238;
    sub_9C6650(&v17);
    *(_QWORD *)v4 = &unk_4A23FE8;
    *(_QWORD *)(v4 + 40) = &unk_4A24030;
    *(_QWORD *)(v4 + 96) = &unk_4A24068;
    sub_9C6650(&v16);
    *(_QWORD *)v4 = &unk_4A24DB8;
    *(_QWORD *)(v4 + 40) = &unk_4A24E00;
    *(_QWORD *)(v4 + 96) = &unk_4A24E38;
  }
  sub_9C6650(&v15);
  if ( *(_DWORD *)(a1 + 56) == 2 )
    sub_2AAECA0(v4 + 40, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), v10, v11, v12, v13);
  return v4;
}
