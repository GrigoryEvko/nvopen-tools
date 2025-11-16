// Function: sub_2C1ACE0
// Address: 0x2c1ace0
//
__int64 __fastcall sub_2C1ACE0(__int64 a1)
{
  int v1; // r14d
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r12
  int v8; // eax
  char *v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  int v17; // [rsp+Ch] [rbp-64h]
  int v18; // [rsp+14h] [rbp-5Ch] BYREF
  __int64 v19; // [rsp+18h] [rbp-58h] BYREF
  __int64 v20; // [rsp+20h] [rbp-50h] BYREF
  __int64 v21; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v22[2]; // [rsp+30h] [rbp-40h] BYREF
  char v23; // [rsp+40h] [rbp-30h] BYREF

  v1 = 0;
  v2 = *(__int64 **)(a1 + 48);
  v3 = *v2;
  v4 = v2[1];
  if ( *(_BYTE *)(a1 + 152) == 5 )
    v1 = sub_2C1A110(a1);
  v7 = sub_22077B0(0xA8u);
  if ( v7 )
  {
    v8 = *(_DWORD *)(a1 + 160);
    v18 = v1;
    v9 = (char *)v22;
    v22[1] = v4;
    v10 = v7 + 64;
    v17 = v8;
    v19 = 0;
    v22[0] = v3;
    *(_QWORD *)v7 = &unk_4A231A8;
    v20 = 0;
    *(_BYTE *)(v7 + 8) = 11;
    *(_QWORD *)(v7 + 40) = &unk_4A23170;
    *(_QWORD *)(v7 + 56) = 0x200000000LL;
    v11 = 0;
    v21 = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 16) = 0;
    for ( *(_QWORD *)(v7 + 48) = v7 + 64; ; v10 = *(_QWORD *)(v7 + 48) )
    {
      *(_QWORD *)(v10 + 8 * v11) = v3;
      ++*(_DWORD *)(v7 + 56);
      v12 = *(unsigned int *)(v3 + 24);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v3 + 28) )
      {
        sub_C8D5F0(v3 + 16, (const void *)(v3 + 32), v12 + 1, 8u, v5, v6);
        v12 = *(unsigned int *)(v3 + 24);
      }
      v9 += 8;
      *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8 * v12) = v7 + 40;
      ++*(_DWORD *)(v3 + 24);
      if ( v9 == &v23 )
        break;
      v11 = *(unsigned int *)(v7 + 56);
      v3 = *(_QWORD *)v9;
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 60) )
      {
        sub_C8D5F0(v7 + 48, (const void *)(v7 + 64), v11 + 1, 8u, v5, v6);
        v11 = *(unsigned int *)(v7 + 56);
      }
    }
    *(_QWORD *)(v7 + 80) = 0;
    *(_QWORD *)(v7 + 40) = &unk_4A23AA8;
    v13 = v21;
    *(_QWORD *)v7 = &unk_4A23A70;
    *(_QWORD *)(v7 + 88) = v13;
    if ( v13 )
      sub_2AAAFA0((__int64 *)(v7 + 88));
    sub_9C6650(&v21);
    sub_2BF0340(v7 + 96, 1, 0, v7, v14, v15);
    *(_QWORD *)v7 = &unk_4A231C8;
    *(_QWORD *)(v7 + 40) = &unk_4A23200;
    *(_QWORD *)(v7 + 96) = &unk_4A23238;
    sub_9C6650(&v20);
    *(_BYTE *)(v7 + 152) = 5;
    *(_QWORD *)v7 = &unk_4A23258;
    *(_QWORD *)(v7 + 40) = &unk_4A23290;
    *(_QWORD *)(v7 + 96) = &unk_4A232C8;
    sub_2C1AC80((_BYTE *)(v7 + 156), &v18);
    sub_9C6650(&v19);
    *(_QWORD *)v7 = &unk_4A24130;
    *(_QWORD *)(v7 + 96) = &unk_4A241A8;
    *(_QWORD *)(v7 + 40) = &unk_4A24170;
    *(_DWORD *)(v7 + 160) = v17;
  }
  return v7;
}
