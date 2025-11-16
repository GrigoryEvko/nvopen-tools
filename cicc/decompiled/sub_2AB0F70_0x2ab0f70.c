// Function: sub_2AB0F70
// Address: 0x2ab0f70
//
__int64 __fastcall sub_2AB0F70(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, void **a5)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // r13
  const void *v12; // rcx
  __int64 v13; // r15
  char *v14; // r14
  __int64 v15; // r12
  const void *v16; // rbx
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v27; // [rsp+40h] [rbp-60h] BYREF
  __int64 v28; // [rsp+48h] [rbp-58h] BYREF
  __int64 v29; // [rsp+50h] [rbp-50h] BYREF
  __int64 v30; // [rsp+58h] [rbp-48h] BYREF
  _QWORD v31[2]; // [rsp+60h] [rbp-40h] BYREF
  char v32; // [rsp+70h] [rbp-30h] BYREF

  v6 = *a4;
  v31[0] = a2;
  v31[1] = a3;
  v27 = v6;
  if ( v6 )
    sub_2AAAFA0(&v27);
  v8 = sub_22077B0(0xC8u);
  if ( !v8 )
  {
    v20 = *a1;
    if ( !*a1 )
      goto LABEL_19;
    v21 = *a1;
    v20 = 96;
    goto LABEL_18;
  }
  v28 = v27;
  if ( !v27 )
  {
    v29 = 0;
    goto LABEL_21;
  }
  sub_2AAAFA0(&v28);
  v29 = v28;
  if ( !v28 )
  {
LABEL_21:
    v30 = 0;
    goto LABEL_8;
  }
  sub_2AAAFA0(&v29);
  v30 = v29;
  if ( v29 )
    sub_2AAAFA0(&v30);
LABEL_8:
  v9 = 0;
  v10 = v8 + 48;
  *(_QWORD *)(v8 + 48) = v8 + 64;
  v11 = v8;
  v12 = (const void *)(v8 + 64);
  *(_BYTE *)(v8 + 8) = 4;
  *(_QWORD *)(v8 + 24) = 0;
  v13 = v8 + 40;
  *(_QWORD *)v8 = &unk_4A231A8;
  *(_QWORD *)(v8 + 32) = 0;
  *(_QWORD *)(v8 + 40) = &unk_4A23170;
  *(_QWORD *)(v8 + 56) = 0x200000000LL;
  *(_QWORD *)(v8 + 16) = 0;
  v14 = (char *)v31;
  v15 = a2;
  v16 = v12;
  while ( 1 )
  {
    *((_QWORD *)v12 + v9) = v15;
    v17 = *(unsigned int *)(v15 + 24);
    v18 = *(unsigned int *)(v15 + 28);
    ++*(_DWORD *)(v11 + 56);
    if ( v17 + 1 > v18 )
    {
      sub_C8D5F0(v15 + 16, (const void *)(v15 + 32), v17 + 1, 8u, v7, v17 + 1);
      v17 = *(unsigned int *)(v15 + 24);
    }
    v14 += 8;
    *(_QWORD *)(*(_QWORD *)(v15 + 16) + 8 * v17) = v13;
    ++*(_DWORD *)(v15 + 24);
    if ( v14 == &v32 )
      break;
    v9 = *(unsigned int *)(v11 + 56);
    v15 = *(_QWORD *)v14;
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 60) )
    {
      sub_C8D5F0(v10, v16, v9 + 1, 8u, v7, v9 + 1);
      v9 = *(unsigned int *)(v11 + 56);
    }
    v12 = *(const void **)(v11 + 48);
  }
  v8 = v11;
  *(_QWORD *)(v11 + 80) = 0;
  *(_QWORD *)(v11 + 40) = &unk_4A23AA8;
  v19 = v30;
  *(_QWORD *)v11 = &unk_4A23A70;
  *(_QWORD *)(v11 + 88) = v19;
  if ( v19 )
    sub_2AAAFA0((__int64 *)(v11 + 88));
  v20 = v11 + 96;
  sub_9C6650(&v30);
  sub_2BF0340(v11 + 96, 1, 0, v11);
  *(_QWORD *)v11 = &unk_4A231C8;
  *(_QWORD *)(v11 + 40) = &unk_4A23200;
  *(_QWORD *)(v11 + 96) = &unk_4A23238;
  sub_9C6650(&v29);
  *(_BYTE *)(v11 + 156) &= ~1u;
  *(_BYTE *)(v11 + 152) = 2;
  *(_QWORD *)v11 = &unk_4A23258;
  *(_QWORD *)(v11 + 40) = &unk_4A23290;
  *(_QWORD *)(v11 + 96) = &unk_4A232C8;
  sub_9C6650(&v28);
  *(_BYTE *)(v11 + 160) = 29;
  *(_QWORD *)v11 = &unk_4A23B70;
  *(_QWORD *)(v11 + 96) = &unk_4A23BF0;
  *(_QWORD *)(v11 + 40) = &unk_4A23BB8;
  sub_CA0F50((__int64 *)(v11 + 168), a5);
  v21 = *a1;
  if ( *a1 )
  {
LABEL_18:
    *(_QWORD *)(v8 + 80) = v21;
    v22 = (__int64 *)a1[1];
    v23 = *(_QWORD *)(v8 + 24) & 7LL;
    v24 = *v22;
    *(_QWORD *)(v8 + 32) = v22;
    v24 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 24) = v24 | v23;
    *(_QWORD *)(v24 + 8) = v8 + 24;
    *v22 = *v22 & 7 | (v8 + 24);
  }
LABEL_19:
  sub_9C6650(&v27);
  return v20;
}
