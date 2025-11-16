// Function: sub_2AAFFE0
// Address: 0x2aaffe0
//
__int64 __fastcall sub_2AAFFE0(_QWORD *a1, char a2, __int64 *a3, __int64 a4, __int64 *a5, void **a6)
{
  __int64 v8; // r8
  __int64 v9; // r15
  __int64 v10; // r9
  __int64 *v11; // rax
  __int64 v12; // r12
  __int64 v13; // rcx
  __int64 *v14; // rbx
  __int64 v15; // rdx
  __int64 *v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v24; // [rsp+50h] [rbp-50h] BYREF
  __int64 v25; // [rsp+58h] [rbp-48h] BYREF
  __int64 v26; // [rsp+60h] [rbp-40h] BYREF
  __int64 v27[7]; // [rsp+68h] [rbp-38h] BYREF

  v24 = *a5;
  if ( v24 )
    sub_2AAAFA0(&v24);
  v9 = sub_22077B0(0xC8u);
  if ( v9 )
  {
    v25 = v24;
    if ( v24 )
    {
      sub_2AAAFA0(&v25);
      v26 = v25;
      if ( v25 )
      {
        sub_2AAAFA0(&v26);
        v27[0] = v26;
        if ( v26 )
          sub_2AAAFA0(v27);
        goto LABEL_8;
      }
    }
    else
    {
      v26 = 0;
    }
    v27[0] = 0;
LABEL_8:
    *(_QWORD *)(v9 + 24) = 0;
    v10 = v9 + 40;
    *(_QWORD *)(v9 + 32) = 0;
    *(_BYTE *)(v9 + 8) = 4;
    *(_QWORD *)v9 = &unk_4A231A8;
    *(_QWORD *)(v9 + 16) = 0;
    *(_QWORD *)(v9 + 48) = v9 + 64;
    *(_QWORD *)(v9 + 40) = &unk_4A23170;
    *(_QWORD *)(v9 + 56) = 0x200000000LL;
    v11 = &a3[a4];
    if ( v11 != a3 )
    {
      v12 = *a3;
      v13 = v9 + 64;
      v14 = a3 + 1;
      v15 = 0;
      v16 = v11;
      while ( 1 )
      {
        *(_QWORD *)(v13 + 8 * v15) = v12;
        ++*(_DWORD *)(v9 + 56);
        v17 = *(unsigned int *)(v12 + 24);
        if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 28) )
        {
          sub_C8D5F0(v12 + 16, (const void *)(v12 + 32), v17 + 1, 8u, v8, v10);
          v17 = *(unsigned int *)(v12 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v12 + 16) + 8 * v17) = v9 + 40;
        ++*(_DWORD *)(v12 + 24);
        if ( v16 == v14 )
          break;
        v15 = *(unsigned int *)(v9 + 56);
        v12 = *v14;
        if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 60) )
        {
          sub_C8D5F0(v9 + 48, (const void *)(v9 + 64), v15 + 1, 8u, v8, v10);
          v15 = *(unsigned int *)(v9 + 56);
        }
        v13 = *(_QWORD *)(v9 + 48);
        ++v14;
      }
    }
    *(_QWORD *)(v9 + 80) = 0;
    *(_QWORD *)(v9 + 40) = &unk_4A23AA8;
    v18 = v27[0];
    *(_QWORD *)v9 = &unk_4A23A70;
    *(_QWORD *)(v9 + 88) = v18;
    if ( v18 )
      sub_2AAAFA0((__int64 *)(v9 + 88));
    sub_9C6650(v27);
    sub_2BF0340(v9 + 96, 1, 0, v9);
    *(_QWORD *)v9 = &unk_4A231C8;
    *(_QWORD *)(v9 + 40) = &unk_4A23200;
    *(_QWORD *)(v9 + 96) = &unk_4A23238;
    sub_9C6650(&v26);
    *(_BYTE *)(v9 + 152) = 7;
    *(_DWORD *)(v9 + 156) = 0;
    *(_QWORD *)v9 = &unk_4A23258;
    *(_QWORD *)(v9 + 40) = &unk_4A23290;
    *(_QWORD *)(v9 + 96) = &unk_4A232C8;
    sub_9C6650(&v25);
    *(_QWORD *)v9 = &unk_4A23B70;
    *(_QWORD *)(v9 + 96) = &unk_4A23BF0;
    *(_QWORD *)(v9 + 40) = &unk_4A23BB8;
    *(_BYTE *)(v9 + 160) = a2;
    sub_CA0F50((__int64 *)(v9 + 168), a6);
  }
  if ( *a1 )
  {
    v19 = (__int64 *)a1[1];
    *(_QWORD *)(v9 + 80) = *a1;
    v20 = *(_QWORD *)(v9 + 24);
    v21 = *v19;
    *(_QWORD *)(v9 + 32) = v19;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v9 + 24) = v21 | v20 & 7;
    *(_QWORD *)(v21 + 8) = v9 + 24;
    *v19 = *v19 & 7 | (v9 + 24);
  }
  sub_9C6650(&v24);
  return v9;
}
