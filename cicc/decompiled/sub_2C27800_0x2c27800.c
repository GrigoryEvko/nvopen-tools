// Function: sub_2C27800
// Address: 0x2c27800
//
__int64 __fastcall sub_2C27800(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int8 *a5,
        unsigned __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  void *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v18; // rax
  __int64 v22; // [rsp+28h] [rbp-48h] BYREF
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  __int64 v24[7]; // [rsp+38h] [rbp-38h] BYREF

  v22 = *a4;
  if ( !v22 )
  {
    v23 = 0;
    goto LABEL_17;
  }
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
  *(_QWORD *)(a1 + 32) = 0;
  *(_BYTE *)(a1 + 8) = 35;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 64) = a2;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000001LL;
  v7 = *(unsigned int *)(a2 + 24);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 28) )
  {
    sub_C8D5F0(a2 + 16, (const void *)(a2 + 32), v7 + 1, 8u, (__int64)a5, a6);
    v7 = *(unsigned int *)(a2 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v7) = a1 + 40;
  ++*(_DWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v8 = v24[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = v8;
  if ( v8 )
    sub_2C25AB0((__int64 *)(a1 + 88));
  sub_9C6650(v24);
  sub_2BF0340(a1 + 96, 1, 0, a1, v9, v10);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v23);
  *(_QWORD *)a1 = &unk_4A23FE8;
  *(_QWORD *)(a1 + 40) = &unk_4A24030;
  *(_QWORD *)(a1 + 96) = &unk_4A24068;
  sub_9C6650(&v22);
  v13 = a1 + 152;
  v14 = (void *)(a1 + 168);
  *(_QWORD *)a1 = &unk_4A24E58;
  v15 = (__int64)&unk_4A24EA8;
  *(_QWORD *)(a1 + 40) = &unk_4A24EA8;
  *(_QWORD *)(a1 + 96) = &unk_4A24EE0;
  if ( !a5 )
  {
    *(_QWORD *)(a1 + 152) = v14;
    *(_QWORD *)(a1 + 160) = 0;
    *(_BYTE *)(a1 + 168) = 0;
    return sub_2AAECA0(a1 + 40, a3, v15, v11, v12, v13);
  }
  *(_QWORD *)(a1 + 152) = v14;
  v24[0] = a6;
  v16 = a6;
  if ( a6 > 0xF )
  {
    v18 = sub_22409D0(a1 + 152, (unsigned __int64 *)v24, 0);
    *(_QWORD *)(a1 + 152) = v18;
    v14 = (void *)v18;
    *(_QWORD *)(a1 + 168) = v24[0];
    goto LABEL_21;
  }
  if ( a6 != 1 )
  {
    if ( !a6 )
      goto LABEL_13;
LABEL_21:
    memcpy(v14, a5, a6);
    v16 = v24[0];
    v14 = *(void **)(a1 + 152);
    goto LABEL_13;
  }
  v15 = *a5;
  *(_BYTE *)(a1 + 168) = v15;
LABEL_13:
  *(_QWORD *)(a1 + 160) = v16;
  *((_BYTE *)v14 + v16) = 0;
  return sub_2AAECA0(a1 + 40, a3, v15, v11, v12, v13);
}
