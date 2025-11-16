// Function: sub_2C27C50
// Address: 0x2c27c50
//
__int64 __fastcall sub_2C27C50(_QWORD *a1, char a2, __int64 *a3, __int64 a4, __int64 *a5, void **a6)
{
  __int64 v8; // r15
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 *v13; // rbx
  __int64 v14; // rcx
  __int64 *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v25; // [rsp+58h] [rbp-58h] BYREF
  __int64 v26; // [rsp+60h] [rbp-50h] BYREF
  __int64 v27; // [rsp+68h] [rbp-48h] BYREF
  __int64 v28; // [rsp+70h] [rbp-40h] BYREF
  __int64 v29[7]; // [rsp+78h] [rbp-38h] BYREF

  v25 = *a5;
  if ( v25 )
  {
    sub_2C25AB0(&v25);
    v26 = v25;
    if ( v25 )
      sub_2C25AB0(&v26);
  }
  else
  {
    v26 = 0;
  }
  v8 = sub_22077B0(0xC8u);
  if ( v8 )
  {
    v27 = v26;
    if ( v26 )
    {
      sub_2C25AB0(&v27);
      v28 = v27;
      if ( v27 )
      {
        sub_2C25AB0(&v28);
        v29[0] = v28;
        if ( v28 )
          sub_2C25AB0(v29);
        goto LABEL_9;
      }
    }
    else
    {
      v28 = 0;
    }
    v29[0] = 0;
LABEL_9:
    v9 = (__int64)&a3[a4];
    *(_QWORD *)(v8 + 24) = 0;
    v10 = v8 + 40;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)v8 = &unk_4A231A8;
    *(_BYTE *)(v8 + 8) = 4;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 40) = &unk_4A23170;
    *(_QWORD *)(v8 + 48) = v8 + 64;
    *(_QWORD *)(v8 + 56) = 0x200000000LL;
    if ( (__int64 *)v9 != a3 )
    {
      v11 = *a3;
      v12 = 0;
      v13 = a3 + 1;
      v14 = v8 + 64;
      v15 = (__int64 *)v9;
      while ( 1 )
      {
        *(_QWORD *)(v14 + 8 * v12) = v11;
        ++*(_DWORD *)(v8 + 56);
        v16 = *(unsigned int *)(v11 + 24);
        if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 28) )
        {
          sub_C8D5F0(v11 + 16, (const void *)(v11 + 32), v16 + 1, 8u, v9, v10);
          v16 = *(unsigned int *)(v11 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v11 + 16) + 8 * v16) = v8 + 40;
        ++*(_DWORD *)(v11 + 24);
        if ( v15 == v13 )
          break;
        v12 = *(unsigned int *)(v8 + 56);
        v11 = *v13;
        if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 60) )
        {
          sub_C8D5F0(v8 + 48, (const void *)(v8 + 64), v12 + 1, 8u, v9, v10);
          v12 = *(unsigned int *)(v8 + 56);
        }
        v14 = *(_QWORD *)(v8 + 48);
        ++v13;
      }
    }
    *(_QWORD *)(v8 + 80) = 0;
    *(_QWORD *)(v8 + 40) = &unk_4A23AA8;
    v17 = v29[0];
    *(_QWORD *)v8 = &unk_4A23A70;
    *(_QWORD *)(v8 + 88) = v17;
    if ( v17 )
      sub_2C25AB0((__int64 *)(v8 + 88));
    sub_9C6650(v29);
    sub_2BF0340(v8 + 96, 1, 0, v8, v18, v19);
    *(_QWORD *)v8 = &unk_4A231C8;
    *(_QWORD *)(v8 + 40) = &unk_4A23200;
    *(_QWORD *)(v8 + 96) = &unk_4A23238;
    sub_9C6650(&v28);
    *(_BYTE *)(v8 + 152) = 7;
    *(_DWORD *)(v8 + 156) = 0;
    *(_QWORD *)v8 = &unk_4A23258;
    *(_QWORD *)(v8 + 40) = &unk_4A23290;
    *(_QWORD *)(v8 + 96) = &unk_4A232C8;
    sub_9C6650(&v27);
    *(_BYTE *)(v8 + 160) = a2;
    *(_QWORD *)v8 = &unk_4A23B70;
    *(_QWORD *)(v8 + 40) = &unk_4A23BB8;
    *(_QWORD *)(v8 + 96) = &unk_4A23BF0;
    sub_CA0F50((__int64 *)(v8 + 168), a6);
  }
  if ( *a1 )
  {
    v20 = (__int64 *)a1[1];
    *(_QWORD *)(v8 + 80) = *a1;
    v21 = *(_QWORD *)(v8 + 24);
    v22 = *v20;
    *(_QWORD *)(v8 + 32) = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 24) = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = v8 + 24;
    *v20 = *v20 & 7 | (v8 + 24);
  }
  sub_9C6650(&v26);
  sub_9C6650(&v25);
  return v8;
}
