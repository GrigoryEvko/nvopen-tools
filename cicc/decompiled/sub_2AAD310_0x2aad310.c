// Function: sub_2AAD310
// Address: 0x2aad310
//
__int64 __fastcall sub_2AAD310(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // r15
  __int64 v5; // r10
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r10
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r10
  __int64 (__fastcall *v12)(__int64); // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+18h] [rbp-58h]
  __int64 v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+18h] [rbp-58h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+28h] [rbp-48h] BYREF
  __int64 v25; // [rsp+30h] [rbp-40h] BYREF
  __int64 v26[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = **(_QWORD **)(a1 + 48);
  v23 = *(_QWORD *)(a1 + 88);
  if ( v23 )
    sub_2AAAFA0(&v23);
  v4 = sub_22077B0(0x98u);
  if ( !v4 )
  {
    v8 = 40;
    goto LABEL_13;
  }
  v24 = v23;
  if ( v23 )
  {
    sub_2AAAFA0(&v24);
    v25 = v24;
    if ( v24 )
    {
      sub_2AAAFA0(&v25);
      v26[0] = v25;
      if ( v25 )
        sub_2AAAFA0(v26);
      goto LABEL_8;
    }
  }
  else
  {
    v25 = 0;
  }
  v26[0] = 0;
LABEL_8:
  *(_BYTE *)(v4 + 8) = 29;
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
  v7 = v26[0];
  *(_QWORD *)v4 = &unk_4A23A70;
  *(_QWORD *)(v4 + 88) = v7;
  if ( v7 )
  {
    v18 = v5;
    sub_2AAAFA0((__int64 *)(v4 + 88));
    v5 = v18;
  }
  v19 = v5;
  sub_9C6650(v26);
  sub_2BF0340(v4 + 96, 1, 0, v4);
  *(_QWORD *)v4 = &unk_4A231C8;
  *(_QWORD *)(v4 + 40) = &unk_4A23200;
  *(_QWORD *)(v4 + 96) = &unk_4A23238;
  sub_9C6650(&v25);
  *(_QWORD *)v4 = &unk_4A23FE8;
  *(_QWORD *)(v4 + 40) = &unk_4A24030;
  *(_QWORD *)(v4 + 96) = &unk_4A24068;
  sub_9C6650(&v24);
  v8 = v19;
  *(_QWORD *)v4 = &unk_4A23390;
  *(_QWORD *)(v4 + 40) = &unk_4A233E8;
  *(_QWORD *)(v4 + 96) = &unk_4A23420;
LABEL_13:
  v20 = v8;
  sub_9C6650(&v23);
  v11 = v20;
  v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL);
  if ( v12 == sub_2AA7530 )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
  }
  else
  {
    v17 = v12(a1);
    v11 = v20;
    v13 = v17;
  }
  v14 = *(unsigned int *)(v4 + 56);
  if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 60) )
  {
    v22 = v11;
    sub_C8D5F0(v4 + 48, (const void *)(v4 + 64), v14 + 1, 8u, v9, v10);
    v14 = *(unsigned int *)(v4 + 56);
    v11 = v22;
  }
  *(_QWORD *)(*(_QWORD *)(v4 + 48) + 8 * v14) = v13;
  ++*(_DWORD *)(v4 + 56);
  v15 = *(unsigned int *)(v13 + 24);
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 28) )
  {
    v21 = v11;
    sub_C8D5F0(v13 + 16, (const void *)(v13 + 32), v15 + 1, 8u, v9, v10);
    v15 = *(unsigned int *)(v13 + 24);
    v11 = v21;
  }
  *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8 * v15) = v11;
  ++*(_DWORD *)(v13 + 24);
  return v4;
}
