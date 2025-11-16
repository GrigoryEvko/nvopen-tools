// Function: sub_2AB1320
// Address: 0x2ab1320
//
__int64 __fastcall sub_2AB1320(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, void **a5)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // rsi
  const void *v11; // rcx
  __int64 v12; // r13
  __int64 v13; // r12
  __int64 v14; // rbx
  const void *v15; // r14
  char *v16; // r15
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v27; // [rsp+48h] [rbp-68h] BYREF
  __int64 v28; // [rsp+50h] [rbp-60h] BYREF
  __int64 v29; // [rsp+58h] [rbp-58h] BYREF
  __int64 v30; // [rsp+60h] [rbp-50h] BYREF
  __int64 v31; // [rsp+68h] [rbp-48h] BYREF
  _QWORD v32[2]; // [rsp+70h] [rbp-40h] BYREF
  char v33; // [rsp+80h] [rbp-30h] BYREF

  v6 = *a4;
  v32[0] = a2;
  v32[1] = a3;
  v27 = v6;
  if ( v6 )
    sub_2AAAFA0(&v27);
  v7 = sub_22077B0(0xC8u);
  if ( !v7 )
  {
    v20 = *a1;
    if ( !*a1 )
      goto LABEL_20;
    v21 = *a1;
    v20 = 96;
    goto LABEL_19;
  }
  v28 = v27;
  if ( !v27 )
  {
    v29 = 0;
    goto LABEL_22;
  }
  sub_2AAAFA0(&v28);
  v29 = v28;
  if ( !v28 )
  {
LABEL_22:
    v30 = 0;
    goto LABEL_23;
  }
  sub_2AAAFA0(&v29);
  v30 = v29;
  if ( !v29 )
  {
LABEL_23:
    v31 = 0;
    goto LABEL_9;
  }
  sub_2AAAFA0(&v30);
  v31 = v30;
  if ( v30 )
    sub_2AAAFA0(&v31);
LABEL_9:
  *(_BYTE *)(v7 + 8) = 4;
  v9 = 0;
  v10 = v7 + 48;
  *(_QWORD *)(v7 + 24) = 0;
  v11 = (const void *)(v7 + 64);
  *(_QWORD *)(v7 + 32) = 0;
  *(_QWORD *)(v7 + 16) = 0;
  *(_QWORD *)v7 = &unk_4A231A8;
  *(_QWORD *)(v7 + 48) = v7 + 64;
  v12 = v7 + 40;
  *(_QWORD *)(v7 + 40) = &unk_4A23170;
  *(_QWORD *)(v7 + 56) = 0x200000000LL;
  v13 = v7;
  v14 = a2;
  v15 = v11;
  v16 = (char *)v32;
  while ( 1 )
  {
    *((_QWORD *)v11 + v9) = v14;
    v17 = *(unsigned int *)(v14 + 24);
    v18 = *(unsigned int *)(v14 + 28);
    ++*(_DWORD *)(v13 + 56);
    if ( v17 + 1 > v18 )
    {
      sub_C8D5F0(v14 + 16, (const void *)(v14 + 32), v17 + 1, 8u, v8, v17 + 1);
      v17 = *(unsigned int *)(v14 + 24);
    }
    v16 += 8;
    *(_QWORD *)(*(_QWORD *)(v14 + 16) + 8 * v17) = v12;
    ++*(_DWORD *)(v14 + 24);
    if ( v16 == &v33 )
      break;
    v9 = *(unsigned int *)(v13 + 56);
    v14 = *(_QWORD *)v16;
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 60) )
    {
      sub_C8D5F0(v10, v15, v9 + 1, 8u, v8, v9 + 1);
      v9 = *(unsigned int *)(v13 + 56);
    }
    v11 = *(const void **)(v13 + 48);
  }
  v7 = v13;
  *(_QWORD *)(v13 + 80) = 0;
  *(_QWORD *)(v13 + 40) = &unk_4A23AA8;
  v19 = v31;
  *(_QWORD *)v13 = &unk_4A23A70;
  *(_QWORD *)(v13 + 88) = v19;
  if ( v19 )
    sub_2AAAFA0((__int64 *)(v13 + 88));
  v20 = v13 + 96;
  sub_9C6650(&v31);
  sub_2BF0340(v13 + 96, 1, 0, v13);
  *(_QWORD *)v13 = &unk_4A231C8;
  *(_QWORD *)(v13 + 40) = &unk_4A23200;
  *(_QWORD *)(v13 + 96) = &unk_4A23238;
  sub_9C6650(&v30);
  *(_BYTE *)(v13 + 152) = 7;
  *(_DWORD *)(v13 + 156) = 0;
  *(_QWORD *)v13 = &unk_4A23258;
  *(_QWORD *)(v13 + 40) = &unk_4A23290;
  *(_QWORD *)(v13 + 96) = &unk_4A232C8;
  sub_9C6650(&v29);
  *(_BYTE *)(v13 + 160) = 83;
  *(_QWORD *)v13 = &unk_4A23B70;
  *(_QWORD *)(v13 + 96) = &unk_4A23BF0;
  *(_QWORD *)(v13 + 40) = &unk_4A23BB8;
  sub_CA0F50((__int64 *)(v13 + 168), a5);
  sub_9C6650(&v28);
  v21 = *a1;
  if ( *a1 )
  {
LABEL_19:
    *(_QWORD *)(v7 + 80) = v21;
    v22 = (__int64 *)a1[1];
    v23 = *(_QWORD *)(v7 + 24) & 7LL;
    v24 = *v22;
    *(_QWORD *)(v7 + 32) = v22;
    v24 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v7 + 24) = v24 | v23;
    *(_QWORD *)(v24 + 8) = v7 + 24;
    *v22 = *v22 & 7 | (v7 + 24);
  }
LABEL_20:
  sub_9C6650(&v27);
  return v20;
}
