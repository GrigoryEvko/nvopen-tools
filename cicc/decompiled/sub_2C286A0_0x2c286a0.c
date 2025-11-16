// Function: sub_2C286A0
// Address: 0x2c286a0
//
__int64 __fastcall sub_2C286A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4, void **a5)
{
  __int64 v5; // r15
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 *v9; // r13
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 v22; // [rsp+40h] [rbp-60h] BYREF
  __int64 v23; // [rsp+48h] [rbp-58h] BYREF
  __int64 v24; // [rsp+50h] [rbp-50h] BYREF
  __int64 v25; // [rsp+58h] [rbp-48h] BYREF
  __int64 v26; // [rsp+60h] [rbp-40h] BYREF
  __int64 v27; // [rsp+68h] [rbp-38h]
  char v28; // [rsp+70h] [rbp-30h] BYREF

  v5 = a2;
  v22 = *a4;
  if ( v22 )
    sub_2C25AB0(&v22);
  v8 = sub_22077B0(0xC8u);
  if ( v8 )
  {
    v23 = v22;
    if ( v22 )
    {
      sub_2C25AB0(&v23);
      v26 = a2;
      v27 = a3;
      v24 = v23;
      if ( v23 )
      {
        sub_2C25AB0(&v24);
        v25 = v24;
        if ( v24 )
          sub_2C25AB0(&v25);
        goto LABEL_8;
      }
    }
    else
    {
      v26 = a2;
      v27 = a3;
      v24 = 0;
    }
    v25 = 0;
LABEL_8:
    v9 = &v26;
    v10 = v8 + 64;
    *(_BYTE *)(v8 + 8) = 4;
    *(_QWORD *)v8 = &unk_4A231A8;
    *(_QWORD *)(v8 + 56) = 0x200000000LL;
    *(_QWORD *)(v8 + 48) = v8 + 64;
    *(_QWORD *)(v8 + 40) = &unk_4A23170;
    v11 = 0;
    *(_QWORD *)(v8 + 24) = 0;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 16) = 0;
    while ( 1 )
    {
      *(_QWORD *)(v10 + 8 * v11) = v5;
      v12 = *(unsigned int *)(v5 + 24);
      v13 = *(unsigned int *)(v5 + 28);
      ++*(_DWORD *)(v8 + 56);
      if ( v12 + 1 > v13 )
      {
        sub_C8D5F0(v5 + 16, (const void *)(v5 + 32), v12 + 1, 8u, v12 + 1, v7);
        v12 = *(unsigned int *)(v5 + 24);
      }
      ++v9;
      *(_QWORD *)(*(_QWORD *)(v5 + 16) + 8 * v12) = v8 + 40;
      ++*(_DWORD *)(v5 + 24);
      if ( v9 == (__int64 *)&v28 )
        break;
      v11 = *(unsigned int *)(v8 + 56);
      v5 = *v9;
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 60) )
      {
        sub_C8D5F0(v8 + 48, (const void *)(v8 + 64), v11 + 1, 8u, v11 + 1, v7);
        v11 = *(unsigned int *)(v8 + 56);
      }
      v10 = *(_QWORD *)(v8 + 48);
    }
    *(_QWORD *)(v8 + 80) = 0;
    *(_QWORD *)(v8 + 40) = &unk_4A23AA8;
    v14 = v25;
    *(_QWORD *)v8 = &unk_4A23A70;
    *(_QWORD *)(v8 + 88) = v14;
    if ( v14 )
      sub_2C25AB0((__int64 *)(v8 + 88));
    sub_9C6650(&v25);
    sub_2BF0340(v8 + 96, 1, 0, v8, v15, v16);
    *(_QWORD *)v8 = &unk_4A231C8;
    *(_QWORD *)(v8 + 40) = &unk_4A23200;
    *(_QWORD *)(v8 + 96) = &unk_4A23238;
    sub_9C6650(&v24);
    *(_BYTE *)(v8 + 152) = 4;
    *(_DWORD *)(v8 + 156) = 0;
    *(_QWORD *)v8 = &unk_4A23258;
    *(_QWORD *)(v8 + 40) = &unk_4A23290;
    *(_QWORD *)(v8 + 96) = &unk_4A232C8;
    sub_9C6650(&v23);
    *(_BYTE *)(v8 + 160) = 84;
    *(_QWORD *)v8 = &unk_4A23B70;
    *(_QWORD *)(v8 + 40) = &unk_4A23BB8;
    *(_QWORD *)(v8 + 96) = &unk_4A23BF0;
    sub_CA0F50((__int64 *)(v8 + 168), a5);
  }
  if ( *a1 )
  {
    v17 = (__int64 *)a1[1];
    *(_QWORD *)(v8 + 80) = *a1;
    v18 = *(_QWORD *)(v8 + 24);
    v19 = *v17;
    *(_QWORD *)(v8 + 32) = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 24) = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v8 + 24;
    *v17 = *v17 & 7 | (v8 + 24);
  }
  sub_9C6650(&v22);
  return v8;
}
