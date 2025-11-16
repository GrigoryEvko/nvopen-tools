// Function: sub_2AB07D0
// Address: 0x2ab07d0
//
__int64 __fastcall sub_2AB07D0(_QWORD *a1, char a2, __int64 *a3, __int64 a4, __int64 a5, void **a6)
{
  __int64 v9; // r8
  __int64 v10; // r14
  __int64 v11; // r9
  __int64 *v12; // rax
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 *v15; // rbx
  __int64 v16; // rcx
  __int64 *v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v25; // [rsp+60h] [rbp-60h] BYREF
  __int64 v26; // [rsp+68h] [rbp-58h] BYREF
  __int64 v27; // [rsp+70h] [rbp-50h] BYREF
  __int64 v28; // [rsp+78h] [rbp-48h] BYREF
  __int64 v29; // [rsp+80h] [rbp-40h] BYREF
  __int64 v30[7]; // [rsp+88h] [rbp-38h] BYREF

  v25 = 0;
  if ( !a5 || (v25 = *(_QWORD *)(a5 + 48)) == 0 )
  {
    v26 = 0;
    goto LABEL_26;
  }
  sub_2AAAFA0(&v25);
  v26 = v25;
  if ( !v25 )
  {
LABEL_26:
    v27 = 0;
    goto LABEL_6;
  }
  sub_2AAAFA0(&v26);
  v27 = v26;
  if ( v26 )
    sub_2AAAFA0(&v27);
LABEL_6:
  v10 = sub_22077B0(0xC8u);
  if ( !v10 )
    goto LABEL_22;
  v28 = v27;
  if ( v27 )
  {
    sub_2AAAFA0(&v28);
    v29 = v28;
    if ( v28 )
    {
      sub_2AAAFA0(&v29);
      v30[0] = v29;
      if ( v29 )
        sub_2AAAFA0(v30);
      goto LABEL_11;
    }
  }
  else
  {
    v29 = 0;
  }
  v30[0] = 0;
LABEL_11:
  *(_QWORD *)(v10 + 24) = 0;
  v11 = v10 + 40;
  *(_QWORD *)(v10 + 32) = 0;
  *(_BYTE *)(v10 + 8) = 4;
  *(_QWORD *)v10 = &unk_4A231A8;
  *(_QWORD *)(v10 + 16) = 0;
  *(_QWORD *)(v10 + 48) = v10 + 64;
  *(_QWORD *)(v10 + 40) = &unk_4A23170;
  *(_QWORD *)(v10 + 56) = 0x200000000LL;
  v12 = &a3[a4];
  if ( v12 != a3 )
  {
    v13 = *a3;
    v14 = 0;
    v15 = a3 + 1;
    v16 = v10 + 64;
    v17 = v12;
    while ( 1 )
    {
      *(_QWORD *)(v16 + 8 * v14) = v13;
      ++*(_DWORD *)(v10 + 56);
      v18 = *(unsigned int *)(v13 + 24);
      if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 28) )
      {
        sub_C8D5F0(v13 + 16, (const void *)(v13 + 32), v18 + 1, 8u, v9, v11);
        v18 = *(unsigned int *)(v13 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8 * v18) = v10 + 40;
      ++*(_DWORD *)(v13 + 24);
      if ( v17 == v15 )
        break;
      v14 = *(unsigned int *)(v10 + 56);
      v13 = *v15;
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 60) )
      {
        sub_C8D5F0(v10 + 48, (const void *)(v10 + 64), v14 + 1, 8u, v9, v11);
        v14 = *(unsigned int *)(v10 + 56);
      }
      v16 = *(_QWORD *)(v10 + 48);
      ++v15;
    }
  }
  *(_QWORD *)(v10 + 80) = 0;
  *(_QWORD *)(v10 + 40) = &unk_4A23AA8;
  v19 = v30[0];
  *(_QWORD *)v10 = &unk_4A23A70;
  *(_QWORD *)(v10 + 88) = v19;
  if ( v19 )
    sub_2AAAFA0((__int64 *)(v10 + 88));
  sub_9C6650(v30);
  sub_2BF0340(v10 + 96, 1, 0, v10);
  *(_QWORD *)v10 = &unk_4A231C8;
  *(_QWORD *)(v10 + 40) = &unk_4A23200;
  *(_QWORD *)(v10 + 96) = &unk_4A23238;
  sub_9C6650(&v29);
  *(_BYTE *)(v10 + 152) = 7;
  *(_DWORD *)(v10 + 156) = 0;
  *(_QWORD *)v10 = &unk_4A23258;
  *(_QWORD *)(v10 + 40) = &unk_4A23290;
  *(_QWORD *)(v10 + 96) = &unk_4A232C8;
  sub_9C6650(&v28);
  *(_QWORD *)v10 = &unk_4A23B70;
  *(_QWORD *)(v10 + 96) = &unk_4A23BF0;
  *(_QWORD *)(v10 + 40) = &unk_4A23BB8;
  *(_BYTE *)(v10 + 160) = a2;
  sub_CA0F50((__int64 *)(v10 + 168), a6);
LABEL_22:
  if ( *a1 )
  {
    v20 = (__int64 *)a1[1];
    *(_QWORD *)(v10 + 80) = *a1;
    v21 = *(_QWORD *)(v10 + 24);
    v22 = *v20;
    *(_QWORD *)(v10 + 32) = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v10 + 24) = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = v10 + 24;
    *v20 = *v20 & 7 | (v10 + 24);
  }
  sub_9C6650(&v27);
  sub_9C6650(&v26);
  *(_QWORD *)(v10 + 136) = a5;
  sub_9C6650(&v25);
  return v10;
}
