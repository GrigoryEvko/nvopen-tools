// Function: sub_2C26D30
// Address: 0x2c26d30
//
__int64 *__fastcall sub_2C26D30(__int64 a1, char a2, __int64 *a3, __int64 a4, __int64 *a5, void **a6)
{
  __int64 *v7; // r12
  __int64 v8; // r9
  __int64 *v9; // r11
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 *v13; // r13
  __int64 *v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v22; // [rsp+48h] [rbp-48h] BYREF
  __int64 v23; // [rsp+50h] [rbp-40h] BYREF
  __int64 v24[7]; // [rsp+58h] [rbp-38h] BYREF

  v22 = *a5;
  if ( !v22 )
  {
    v23 = 0;
    v7 = a3;
    goto LABEL_17;
  }
  v7 = a3;
  sub_2C25AB0(&v22);
  v23 = v22;
  if ( !v22 )
  {
LABEL_17:
    v24[0] = 0;
    goto LABEL_5;
  }
  sub_2C25AB0(&v23);
  v24[0] = v23;
  if ( v23 )
    sub_2C25AB0(v24);
LABEL_5:
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  v8 = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  v9 = &a3[a4];
  *(_BYTE *)(a1 + 8) = 4;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  if ( a3 != v9 )
  {
    v10 = *a3;
    v11 = a1 + 64;
    v12 = 0;
    v13 = v7;
    v14 = v9;
    while ( 1 )
    {
      *(_QWORD *)(v11 + 8 * v12) = v10;
      ++*(_DWORD *)(a1 + 56);
      v15 = *(unsigned int *)(v10 + 24);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 28) )
      {
        sub_C8D5F0(v10 + 16, (const void *)(v10 + 32), v15 + 1, 8u, v15 + 1, v8);
        v15 = *(unsigned int *)(v10 + 24);
      }
      ++v13;
      *(_QWORD *)(*(_QWORD *)(v10 + 16) + 8 * v15) = a1 + 40;
      ++*(_DWORD *)(v10 + 24);
      if ( v14 == v13 )
        break;
      v12 = *(unsigned int *)(a1 + 56);
      v10 = *v13;
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v12 + 1, 8u, v12 + 1, v8);
        v12 = *(unsigned int *)(a1 + 56);
      }
      v11 = *(_QWORD *)(a1 + 48);
    }
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v16 = v24[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v16;
  if ( v16 )
    sub_2C25AB0((__int64 *)(a1 + 88));
  sub_9C6650(v24);
  sub_2BF0340(a1 + 96, 1, 0, a1, v17, v18);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v23);
  *(_BYTE *)(a1 + 152) = 7;
  *(_DWORD *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  sub_9C6650(&v22);
  *(_BYTE *)(a1 + 160) = a2;
  *(_QWORD *)a1 = &unk_4A23B70;
  *(_QWORD *)(a1 + 40) = &unk_4A23BB8;
  *(_QWORD *)(a1 + 96) = &unk_4A23BF0;
  return sub_CA0F50((__int64 *)(a1 + 168), a6);
}
