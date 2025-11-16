// Function: sub_2AB0C10
// Address: 0x2ab0c10
//
__int64 __fastcall sub_2AB0C10(_QWORD *a1, __int64 a2, __int64 *a3, void **a4)
{
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v15; // [rsp+30h] [rbp-60h] BYREF
  __int64 v16; // [rsp+38h] [rbp-58h] BYREF
  __int64 v17; // [rsp+40h] [rbp-50h] BYREF
  __int64 v18; // [rsp+48h] [rbp-48h] BYREF
  __int64 v19; // [rsp+50h] [rbp-40h] BYREF
  __int64 v20[7]; // [rsp+58h] [rbp-38h] BYREF

  v15 = *a3;
  if ( !v15 )
  {
    v16 = 0;
    goto LABEL_21;
  }
  sub_2AAAFA0(&v15);
  v16 = v15;
  if ( !v15 )
  {
LABEL_21:
    v17 = 0;
    goto LABEL_5;
  }
  sub_2AAAFA0(&v16);
  v17 = v16;
  if ( v16 )
    sub_2AAAFA0(&v17);
LABEL_5:
  v6 = sub_22077B0(0xC8u);
  if ( !v6 )
    goto LABEL_16;
  v18 = v17;
  if ( v17 )
  {
    sub_2AAAFA0(&v18);
    v19 = v18;
    if ( v18 )
    {
      sub_2AAAFA0(&v19);
      v20[0] = v19;
      if ( v19 )
        sub_2AAAFA0(v20);
      goto LABEL_10;
    }
  }
  else
  {
    v19 = 0;
  }
  v20[0] = 0;
LABEL_10:
  v7 = *(unsigned int *)(a2 + 24);
  *(_BYTE *)(v6 + 8) = 4;
  v8 = v6 + 40;
  *(_QWORD *)(v6 + 24) = 0;
  *(_QWORD *)(v6 + 32) = 0;
  *(_QWORD *)(v6 + 16) = 0;
  *(_QWORD *)v6 = &unk_4A231A8;
  *(_QWORD *)(v6 + 64) = a2;
  *(_QWORD *)(v6 + 40) = &unk_4A23170;
  *(_QWORD *)(v6 + 48) = v6 + 64;
  *(_QWORD *)(v6 + 56) = 0x200000001LL;
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 28) )
  {
    sub_C8D5F0(a2 + 16, (const void *)(a2 + 32), v7 + 1, 8u, v4, v5);
    v7 = *(unsigned int *)(a2 + 24);
    v8 = v6 + 40;
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v7) = v8;
  ++*(_DWORD *)(a2 + 24);
  *(_QWORD *)(v6 + 80) = 0;
  *(_QWORD *)(v6 + 40) = &unk_4A23AA8;
  v9 = v20[0];
  *(_QWORD *)v6 = &unk_4A23A70;
  *(_QWORD *)(v6 + 88) = v9;
  if ( v9 )
  {
    sub_2AAAFA0((__int64 *)(v6 + 88));
    if ( v20[0] )
      sub_B91220((__int64)v20, v20[0]);
  }
  sub_2BF0340(v6 + 96, 1, 0, v6);
  *(_QWORD *)v6 = &unk_4A231C8;
  *(_QWORD *)(v6 + 40) = &unk_4A23200;
  *(_QWORD *)(v6 + 96) = &unk_4A23238;
  sub_9C6650(&v19);
  *(_BYTE *)(v6 + 152) = 7;
  *(_DWORD *)(v6 + 156) = 0;
  *(_QWORD *)v6 = &unk_4A23258;
  *(_QWORD *)(v6 + 40) = &unk_4A23290;
  *(_QWORD *)(v6 + 96) = &unk_4A232C8;
  sub_9C6650(&v18);
  *(_BYTE *)(v6 + 160) = 70;
  *(_QWORD *)v6 = &unk_4A23B70;
  *(_QWORD *)(v6 + 40) = &unk_4A23BB8;
  *(_QWORD *)(v6 + 96) = &unk_4A23BF0;
  sub_CA0F50((__int64 *)(v6 + 168), a4);
LABEL_16:
  if ( *a1 )
  {
    v10 = (__int64 *)a1[1];
    *(_QWORD *)(v6 + 80) = *a1;
    v11 = *(_QWORD *)(v6 + 24);
    v12 = *v10;
    *(_QWORD *)(v6 + 32) = v10;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v6 + 24) = v12 | v11 & 7;
    *(_QWORD *)(v12 + 8) = v6 + 24;
    *v10 = *v10 & 7 | (v6 + 24);
    sub_9C6650(&v17);
    sub_9C6650(&v16);
  }
  else
  {
    sub_9C6650(&v17);
    sub_9C6650(&v16);
    if ( !v6 )
      goto LABEL_19;
  }
  v6 += 96;
LABEL_19:
  sub_9C6650(&v15);
  return v6;
}
