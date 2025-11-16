// Function: sub_2C27000
// Address: 0x2c27000
//
__int64 *__fastcall sub_2C27000(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 *a5, void **a6)
{
  __int64 v6; // r12
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v20; // [rsp+38h] [rbp-58h] BYREF
  __int64 v21; // [rsp+40h] [rbp-50h] BYREF
  __int64 v22; // [rsp+48h] [rbp-48h] BYREF
  __int64 v23; // [rsp+50h] [rbp-40h] BYREF
  __int64 v24; // [rsp+58h] [rbp-38h]
  char v25; // [rsp+60h] [rbp-30h] BYREF

  v6 = a2;
  v20 = *a5;
  if ( !v20 )
  {
    v23 = a2;
    v24 = a3;
    v21 = 0;
    goto LABEL_16;
  }
  sub_2C25AB0(&v20);
  v23 = a2;
  v24 = a3;
  v21 = v20;
  if ( !v20 )
  {
LABEL_16:
    v22 = 0;
    goto LABEL_5;
  }
  sub_2C25AB0(&v21);
  v22 = v21;
  if ( v21 )
    sub_2C25AB0(&v22);
LABEL_5:
  v8 = a1 + 40;
  v9 = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  v10 = &v23;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  v11 = 0;
  *(_BYTE *)(a1 + 8) = 4;
  *(_QWORD *)(a1 + 16) = 0;
  for ( *(_QWORD *)(a1 + 48) = a1 + 64; ; v9 = *(_QWORD *)(a1 + 48) )
  {
    *(_QWORD *)(v9 + 8 * v11) = v6;
    v12 = *(unsigned int *)(v6 + 24);
    v13 = *(unsigned int *)(v6 + 28);
    ++*(_DWORD *)(a1 + 56);
    if ( v12 + 1 > v13 )
    {
      sub_C8D5F0(v6 + 16, (const void *)(v6 + 32), v12 + 1, 8u, v12 + 1, v8);
      v12 = *(unsigned int *)(v6 + 24);
    }
    ++v10;
    *(_QWORD *)(*(_QWORD *)(v6 + 16) + 8 * v12) = a1 + 40;
    ++*(_DWORD *)(v6 + 24);
    if ( v10 == (__int64 *)&v25 )
      break;
    v11 = *(unsigned int *)(a1 + 56);
    v6 = *v10;
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
    {
      sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v11 + 1, 8u, v11 + 1, v8);
      v11 = *(unsigned int *)(a1 + 56);
    }
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v14 = v22;
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v14;
  if ( v14 )
    sub_2C25AB0((__int64 *)(a1 + 88));
  sub_9C6650(&v22);
  sub_2BF0340(a1 + 96, 1, 0, a1, v15, v16);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v21);
  *(_BYTE *)(a1 + 152) = 4;
  *(_DWORD *)(a1 + 156) = a4;
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  sub_9C6650(&v20);
  *(_BYTE *)(a1 + 160) = 84;
  *(_QWORD *)a1 = &unk_4A23B70;
  *(_QWORD *)(a1 + 40) = &unk_4A23BB8;
  *(_QWORD *)(a1 + 96) = &unk_4A23BF0;
  return sub_CA0F50((__int64 *)(a1 + 168), a6);
}
