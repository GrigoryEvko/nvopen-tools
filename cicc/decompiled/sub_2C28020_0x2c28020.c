// Function: sub_2C28020
// Address: 0x2c28020
//
__int64 __fastcall sub_2C28020(_QWORD *a1, char a2, __int64 *a3, __int64 a4, char a5, __int64 *a6, void **a7)
{
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // rbx
  __int64 v14; // rcx
  __int64 *v15; // r12
  __int64 v16; // rdx
  __int64 *v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v27; // [rsp+40h] [rbp-50h] BYREF
  __int64 v28; // [rsp+48h] [rbp-48h] BYREF
  __int64 v29; // [rsp+50h] [rbp-40h] BYREF
  __int64 v30[7]; // [rsp+58h] [rbp-38h] BYREF

  v27 = *a6;
  if ( v27 )
    sub_2C25AB0(&v27);
  v9 = sub_22077B0(0xC8u);
  if ( v9 )
  {
    v28 = v27;
    if ( v27 )
    {
      sub_2C25AB0(&v28);
      v29 = v28;
      if ( v28 )
      {
        sub_2C25AB0(&v29);
        v30[0] = v29;
        if ( v29 )
          sub_2C25AB0(v30);
        goto LABEL_8;
      }
    }
    else
    {
      v29 = 0;
    }
    v30[0] = 0;
LABEL_8:
    v10 = a4;
    if ( !(v10 * 8) )
      a3 = 0;
    *(_BYTE *)(v9 + 8) = 4;
    v11 = v9 + 40;
    *(_QWORD *)(v9 + 24) = 0;
    v12 = (__int64)&a3[v10];
    *(_QWORD *)(v9 + 48) = v9 + 64;
    *(_QWORD *)v9 = &unk_4A231A8;
    *(_QWORD *)(v9 + 32) = 0;
    *(_QWORD *)(v9 + 16) = 0;
    *(_QWORD *)(v9 + 40) = &unk_4A23170;
    *(_QWORD *)(v9 + 56) = 0x200000000LL;
    if ( a3 != &a3[v10] )
    {
      v13 = *a3;
      v14 = v9 + 64;
      v15 = a3 + 1;
      v16 = 0;
      v17 = (__int64 *)v12;
      while ( 1 )
      {
        *(_QWORD *)(v14 + 8 * v16) = v13;
        ++*(_DWORD *)(v9 + 56);
        v18 = *(unsigned int *)(v13 + 24);
        if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 28) )
        {
          sub_C8D5F0(v13 + 16, (const void *)(v13 + 32), v18 + 1, 8u, v12, v11);
          v18 = *(unsigned int *)(v13 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8 * v18) = v9 + 40;
        ++*(_DWORD *)(v13 + 24);
        if ( v17 == v15 )
          break;
        v16 = *(unsigned int *)(v9 + 56);
        v13 = *v15;
        if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 60) )
        {
          sub_C8D5F0(v9 + 48, (const void *)(v9 + 64), v16 + 1, 8u, v12, v11);
          v16 = *(unsigned int *)(v9 + 56);
        }
        v14 = *(_QWORD *)(v9 + 48);
        ++v15;
      }
    }
    *(_QWORD *)(v9 + 80) = 0;
    *(_QWORD *)(v9 + 40) = &unk_4A23AA8;
    v19 = v30[0];
    *(_QWORD *)v9 = &unk_4A23A70;
    *(_QWORD *)(v9 + 88) = v19;
    if ( v19 )
      sub_2C25AB0((__int64 *)(v9 + 88));
    sub_9C6650(v30);
    sub_2BF0340(v9 + 96, 1, 0, v9, v20, v21);
    *(_QWORD *)v9 = &unk_4A231C8;
    *(_QWORD *)(v9 + 40) = &unk_4A23200;
    *(_QWORD *)(v9 + 96) = &unk_4A23238;
    sub_9C6650(&v29);
    *(_BYTE *)(v9 + 152) = 1;
    *(_BYTE *)(v9 + 156) = a5;
    *(_QWORD *)v9 = &unk_4A23258;
    *(_QWORD *)(v9 + 40) = &unk_4A23290;
    *(_QWORD *)(v9 + 96) = &unk_4A232C8;
    sub_9C6650(&v28);
    *(_BYTE *)(v9 + 160) = a2;
    *(_QWORD *)v9 = &unk_4A23B70;
    *(_QWORD *)(v9 + 40) = &unk_4A23BB8;
    *(_QWORD *)(v9 + 96) = &unk_4A23BF0;
    sub_CA0F50((__int64 *)(v9 + 168), a7);
  }
  if ( *a1 )
  {
    v22 = (__int64 *)a1[1];
    *(_QWORD *)(v9 + 80) = *a1;
    v23 = *(_QWORD *)(v9 + 24);
    v24 = *v22;
    *(_QWORD *)(v9 + 32) = v22;
    v24 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v9 + 24) = v24 | v23 & 7;
    *(_QWORD *)(v24 + 8) = v9 + 24;
    *v22 = *v22 & 7 | (v9 + 24);
  }
  sub_9C6650(&v27);
  return v9;
}
