// Function: sub_2C127E0
// Address: 0x2c127e0
//
__int64 __fastcall sub_2C127E0(__int64 a1)
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
  __int64 v11; // [rsp+10h] [rbp-60h]
  int v12; // [rsp+1Ch] [rbp-54h]
  __int64 v13; // [rsp+20h] [rbp-50h] BYREF
  __int64 v14; // [rsp+28h] [rbp-48h] BYREF
  __int64 v15; // [rsp+30h] [rbp-40h] BYREF
  __int64 v16[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = **(_QWORD **)(a1 + 48);
  v12 = *(_DWORD *)(a1 + 156);
  v13 = *(_QWORD *)(a1 + 88);
  if ( v13 )
    sub_2AAAFA0(&v13);
  v4 = sub_22077B0(0xA8u);
  if ( v4 )
  {
    v11 = *(_QWORD *)(a1 + 160);
    v14 = v13;
    if ( v13 )
    {
      sub_2AAAFA0(&v14);
      v15 = v14;
      if ( v14 )
      {
        sub_2AAAFA0(&v15);
        v16[0] = v15;
        if ( v15 )
          sub_2AAAFA0(v16);
        goto LABEL_8;
      }
    }
    else
    {
      v15 = 0;
    }
    v16[0] = 0;
LABEL_8:
    *(_BYTE *)(v4 + 8) = 12;
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
    v7 = v16[0];
    *(_QWORD *)v4 = &unk_4A23A70;
    *(_QWORD *)(v4 + 88) = v7;
    if ( v7 )
      sub_2AAAFA0((__int64 *)(v4 + 88));
    sub_9C6650(v16);
    sub_2BF0340(v4 + 96, 1, 0, v4, v8, v9);
    *(_QWORD *)v4 = &unk_4A231C8;
    *(_QWORD *)(v4 + 40) = &unk_4A23200;
    *(_QWORD *)(v4 + 96) = &unk_4A23238;
    sub_9C6650(&v15);
    *(_BYTE *)(v4 + 152) = 4;
    *(_QWORD *)v4 = &unk_4A23258;
    *(_QWORD *)(v4 + 96) = &unk_4A232C8;
    *(_QWORD *)(v4 + 40) = &unk_4A23290;
    *(_DWORD *)(v4 + 156) = v12;
    sub_9C6650(&v14);
    *(_QWORD *)v4 = &unk_4A24300;
    *(_QWORD *)(v4 + 96) = &unk_4A24380;
    *(_QWORD *)(v4 + 40) = &unk_4A24348;
    *(_QWORD *)(v4 + 160) = v11;
  }
  sub_9C6650(&v13);
  return v4;
}
