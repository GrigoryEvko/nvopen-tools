// Function: sub_2C1A5F0
// Address: 0x2c1a5f0
//
__int64 *__fastcall sub_2C1A5F0(__int64 a1, char a2, int a3, __int64 a4, __int64 a5, __int64 *a6, void **a7)
{
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r15
  __int64 *v14; // r12
  __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v22; // [rsp+38h] [rbp-58h] BYREF
  __int64 v23; // [rsp+40h] [rbp-50h] BYREF
  __int64 v24; // [rsp+48h] [rbp-48h] BYREF
  __int64 v25; // [rsp+50h] [rbp-40h] BYREF
  __int64 v26; // [rsp+58h] [rbp-38h]
  char v27; // [rsp+60h] [rbp-30h] BYREF

  v22 = *a6;
  if ( !v22 )
  {
    v25 = a4;
    v26 = a5;
    v23 = 0;
    goto LABEL_16;
  }
  sub_2AAAFA0(&v22);
  v25 = a4;
  v26 = a5;
  v23 = v22;
  if ( !v22 )
  {
LABEL_16:
    v24 = 0;
    goto LABEL_5;
  }
  sub_2AAAFA0(&v23);
  v24 = v23;
  if ( v23 )
    sub_2AAAFA0(&v24);
LABEL_5:
  v9 = a1 + 64;
  v10 = a1 + 40;
  *(_BYTE *)(a1 + 8) = 4;
  v11 = 0;
  *(_QWORD *)(a1 + 24) = 0;
  v12 = a1 + 64;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  v13 = a4;
  v14 = &v25;
  while ( 1 )
  {
    *(_QWORD *)(v12 + 8 * v11) = v13;
    v15 = *(unsigned int *)(v13 + 24);
    v16 = *(unsigned int *)(v13 + 28);
    ++*(_DWORD *)(a1 + 56);
    if ( v15 + 1 > v16 )
    {
      sub_C8D5F0(v13 + 16, (const void *)(v13 + 32), v15 + 1, 8u, v10, v9);
      v15 = *(unsigned int *)(v13 + 24);
    }
    ++v14;
    *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8 * v15) = a1 + 40;
    ++*(_DWORD *)(v13 + 24);
    if ( v14 == (__int64 *)&v27 )
      break;
    v11 = *(unsigned int *)(a1 + 56);
    v13 = *v14;
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
    {
      sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v11 + 1, 8u, v10, v9);
      v11 = *(unsigned int *)(a1 + 56);
    }
    v12 = *(_QWORD *)(a1 + 48);
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v17 = v24;
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v17;
  if ( v17 )
    sub_2AAAFA0((__int64 *)(a1 + 88));
  sub_9C6650(&v24);
  sub_2BF0340(a1 + 96, 1, 0, a1, v18, v19);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v23);
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  *(_DWORD *)(a1 + 156) = a3;
  sub_9C6650(&v22);
  *(_QWORD *)a1 = &unk_4A23B70;
  *(_QWORD *)(a1 + 96) = &unk_4A23BF0;
  *(_QWORD *)(a1 + 40) = &unk_4A23BB8;
  *(_BYTE *)(a1 + 160) = a2;
  return sub_CA0F50((__int64 *)(a1 + 168), a7);
}
