// Function: sub_2AAF680
// Address: 0x2aaf680
//
__int64 __fastcall sub_2AAF680(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        void **a6,
        int a7,
        char a8)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v18; // [rsp+20h] [rbp-70h] BYREF
  __int64 v19; // [rsp+28h] [rbp-68h] BYREF
  __int64 v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 v21; // [rsp+38h] [rbp-58h] BYREF
  __int64 v22[10]; // [rsp+40h] [rbp-50h] BYREF

  v9 = *a5;
  v22[0] = a2;
  v22[1] = a3;
  v22[2] = a4;
  if ( a8 )
  {
    v21 = v9;
    if ( v9 )
      sub_2AAAFA0(&v21);
    v10 = sub_22077B0(0xC8u);
    v11 = v10;
    if ( v10 )
    {
      sub_2C1AF80(v10, 57, (unsigned int)v22, 3, a7, (unsigned int)&v21, (__int64)a6);
      sub_9C6650(&v21);
      v12 = *a1;
      if ( !*a1 )
        return v11 + 96;
      goto LABEL_15;
    }
    sub_9C6650(&v21);
    v12 = *a1;
    if ( !*a1 )
      return 0;
LABEL_15:
    v14 = (__int64 *)a1[1];
    *(_QWORD *)(v11 + 80) = v12;
    v15 = *(_QWORD *)(v11 + 24);
    v16 = *v14;
    *(_QWORD *)(v11 + 32) = v14;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v11 + 24) = v16 | v15 & 7;
    *(_QWORD *)(v16 + 8) = v11 + 24;
    *v14 = *v14 & 7 | (v11 + 24);
    return v11 + 96;
  }
  v18 = v9;
  if ( v9 )
    sub_2AAAFA0(&v18);
  v11 = sub_22077B0(0xC8u);
  if ( v11 )
  {
    v19 = v18;
    if ( v18 )
    {
      sub_2AAAFA0(&v19);
      v20 = v19;
      if ( v19 )
      {
        sub_2AAAFA0(&v20);
        v21 = v20;
        if ( v20 )
          sub_2AAAFA0(&v21);
LABEL_14:
        sub_2AAF4A0(v11, 4, v22, 3, &v21, v13);
        sub_9C6650(&v21);
        *(_BYTE *)(v11 + 152) = 7;
        *(_DWORD *)(v11 + 156) = 0;
        *(_QWORD *)v11 = &unk_4A23258;
        *(_QWORD *)(v11 + 40) = &unk_4A23290;
        *(_QWORD *)(v11 + 96) = &unk_4A232C8;
        sub_9C6650(&v20);
        *(_BYTE *)(v11 + 160) = 57;
        *(_QWORD *)v11 = &unk_4A23B70;
        *(_QWORD *)(v11 + 96) = &unk_4A23BF0;
        *(_QWORD *)(v11 + 40) = &unk_4A23BB8;
        sub_CA0F50((__int64 *)(v11 + 168), a6);
        sub_9C6650(&v19);
        sub_9C6650(&v18);
        v12 = *a1;
        if ( !*a1 )
          return v11 + 96;
        goto LABEL_15;
      }
    }
    else
    {
      v20 = 0;
    }
    v21 = 0;
    goto LABEL_14;
  }
  sub_9C6650(&v18);
  if ( *a1 )
  {
    MEMORY[0x50] = *a1;
    BUG();
  }
  return 0;
}
