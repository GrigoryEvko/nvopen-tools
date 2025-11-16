// Function: sub_2C1AF80
// Address: 0x2c1af80
//
__int64 *__fastcall sub_2C1AF80(__int64 a1, char a2, __int64 *a3, __int64 a4, int a5, __int64 *a6, void **a7)
{
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 *v17; // r12
  __int64 *v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  int v24; // [rsp+34h] [rbp-4Ch] BYREF
  __int64 v25; // [rsp+38h] [rbp-48h] BYREF
  __int64 v26; // [rsp+40h] [rbp-40h] BYREF
  __int64 v27[7]; // [rsp+48h] [rbp-38h] BYREF

  v25 = *a6;
  if ( !v25 )
  {
    v24 = a5;
    v26 = 0;
    goto LABEL_19;
  }
  sub_2AAAFA0(&v25);
  v24 = a5;
  v26 = v25;
  if ( !v25 )
  {
LABEL_19:
    v27[0] = 0;
    goto LABEL_5;
  }
  sub_2AAAFA0(&v26);
  v27[0] = v26;
  if ( v26 )
    sub_2AAAFA0(v27);
LABEL_5:
  v10 = a4;
  v11 = 0;
  if ( v10 * 8 )
    v11 = a3;
  *(_BYTE *)(a1 + 8) = 4;
  v12 = a1 + 40;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  v13 = (__int64)&v11[v10];
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  if ( v11 != &v11[v10] )
  {
    v14 = a1 + 64;
    v15 = *v11;
    v16 = 0;
    v17 = (__int64 *)v13;
    v18 = v11;
    while ( 1 )
    {
      *(_QWORD *)(v14 + 8 * v16) = v15;
      ++*(_DWORD *)(a1 + 56);
      v19 = *(unsigned int *)(v15 + 24);
      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 28) )
      {
        sub_C8D5F0(v15 + 16, (const void *)(v15 + 32), v19 + 1, 8u, v13, v12);
        v19 = *(unsigned int *)(v15 + 24);
      }
      ++v18;
      *(_QWORD *)(*(_QWORD *)(v15 + 16) + 8 * v19) = a1 + 40;
      ++*(_DWORD *)(v15 + 24);
      if ( v17 == v18 )
        break;
      v16 = *(unsigned int *)(a1 + 56);
      v15 = *v18;
      if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v16 + 1, 8u, v13, v12);
        v16 = *(unsigned int *)(a1 + 56);
      }
      v14 = *(_QWORD *)(a1 + 48);
    }
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v20 = v27[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v20;
  if ( v20 )
    sub_2AAAFA0((__int64 *)(a1 + 88));
  sub_9C6650(v27);
  sub_2BF0340(a1 + 96, 1, 0, a1, v21, v22);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v26);
  *(_BYTE *)(a1 + 152) = 5;
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  sub_2C1AC80((_BYTE *)(a1 + 156), &v24);
  sub_9C6650(&v25);
  *(_QWORD *)a1 = &unk_4A23B70;
  *(_QWORD *)(a1 + 96) = &unk_4A23BF0;
  *(_QWORD *)(a1 + 40) = &unk_4A23BB8;
  *(_BYTE *)(a1 + 160) = a2;
  return sub_CA0F50((__int64 *)(a1 + 168), a7);
}
