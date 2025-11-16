// Function: sub_27C20B0
// Address: 0x27c20b0
//
__int64 __fastcall sub_27C20B0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // r13
  _QWORD *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // r12
  _QWORD *v14; // r13
  __int64 v15; // rax
  _QWORD *v17; // r12
  _QWORD *v18; // r13
  __int64 v19; // rax
  _QWORD *v20; // r12
  _QWORD *v21; // r13
  __int64 v22; // rax
  _QWORD *v23; // r12
  _QWORD *v24; // r13
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 v28; // rax
  void *v29; // [rsp+0h] [rbp-90h] BYREF
  __int64 v30; // [rsp+8h] [rbp-88h] BYREF
  __int64 v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+18h] [rbp-78h]
  char v33; // [rsp+20h] [rbp-70h]
  void *v34; // [rsp+30h] [rbp-60h] BYREF
  __int64 v35; // [rsp+38h] [rbp-58h] BYREF
  __int64 v36; // [rsp+40h] [rbp-50h]
  __int64 v37; // [rsp+48h] [rbp-48h]
  char v38; // [rsp+50h] [rbp-40h]

  v2 = a1 + 800;
  v3 = *(_QWORD *)(a1 + 784);
  if ( v3 != v2 )
    _libc_free(v3);
  sub_B32BF0((_QWORD *)(a1 + 744));
  *(_QWORD *)(a1 + 648) = &unk_49E5698;
  *(_QWORD *)(a1 + 656) = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  v4 = *(_QWORD *)(a1 + 520);
  if ( v4 != a1 + 536 )
    _libc_free(v4);
  v5 = *(unsigned int *)(a1 + 504);
  if ( (_DWORD)v5 )
  {
    v17 = *(_QWORD **)(a1 + 488);
    v31 = -4096;
    v29 = 0;
    v30 = 0;
    v18 = &v17[3 * v5];
    v34 = 0;
    v35 = 0;
    v36 = -8192;
    do
    {
      v19 = v17[2];
      if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
        sub_BD60C0(v17);
      v17 += 3;
    }
    while ( v18 != v17 );
    if ( v31 != -4096 && v31 != 0 )
      sub_BD60C0(&v29);
    v5 = *(unsigned int *)(a1 + 504);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 488), 24 * v5, 8);
  if ( !*(_BYTE *)(a1 + 444) )
    _libc_free(*(_QWORD *)(a1 + 424));
  sub_C7D6A0(*(_QWORD *)(a1 + 392), 16LL * *(unsigned int *)(a1 + 408), 8);
  v6 = *(_QWORD **)(a1 + 320);
  v7 = &v6[3 * *(unsigned int *)(a1 + 328)];
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(v7 - 1);
      v7 -= 3;
      if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
        sub_BD60C0(v7);
    }
    while ( v6 != v7 );
    v7 = *(_QWORD **)(a1 + 320);
  }
  if ( v7 != (_QWORD *)(a1 + 336) )
    _libc_free((unsigned __int64)v7);
  v9 = *(unsigned int *)(a1 + 312);
  if ( (_DWORD)v9 )
  {
    v26 = *(_QWORD *)(a1 + 296);
    v30 = 2;
    v33 = 0;
    v31 = 0;
    v27 = v26 + 48 * v9;
    v29 = &unk_49E51C0;
    v32 = -4096;
    v35 = 2;
    v36 = 0;
    v34 = &unk_49E51C0;
    v38 = 0;
    v37 = -8192;
    do
    {
      while ( *(_BYTE *)(v26 + 32) )
      {
        *(_QWORD *)(v26 + 24) = 0;
        v26 += 48;
        *(_QWORD *)(v26 - 48) = &unk_49DB368;
        if ( v27 == v26 )
          goto LABEL_74;
      }
      v28 = *(_QWORD *)(v26 + 24);
      *(_QWORD *)v26 = &unk_49DB368;
      if ( v28 != -4096 && v28 != -8192 && v28 )
        sub_BD60C0((_QWORD *)(v26 + 8));
      v26 += 48;
    }
    while ( v27 != v26 );
LABEL_74:
    if ( !v38 )
    {
      v34 = &unk_49DB368;
      if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
        sub_BD60C0(&v35);
    }
    if ( !v33 )
    {
      v29 = &unk_49DB368;
      if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
        sub_BD60C0(&v30);
    }
    v9 = *(unsigned int *)(a1 + 312);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 296), 48 * v9, 8);
  if ( !*(_BYTE *)(a1 + 156) )
    _libc_free(*(_QWORD *)(a1 + 136));
  v10 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v10 )
  {
    v23 = *(_QWORD **)(a1 + 104);
    v31 = -4096;
    v29 = 0;
    v30 = 0;
    v24 = &v23[3 * v10];
    v34 = 0;
    v35 = 0;
    v36 = -8192;
    do
    {
      v25 = v23[2];
      if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
        sub_BD60C0(v23);
      v23 += 3;
    }
    while ( v24 != v23 );
    if ( v36 != -4096 && v36 != 0 && v36 != -8192 )
      sub_BD60C0(&v34);
    if ( v31 != -4096 && v31 != 0 && v31 != -8192 )
      sub_BD60C0(&v29);
    v10 = *(unsigned int *)(a1 + 120);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 24 * v10, 8);
  v11 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v11 )
  {
    v20 = *(_QWORD **)(a1 + 72);
    v31 = -4096;
    v29 = 0;
    v30 = 0;
    v21 = &v20[3 * v11];
    v34 = 0;
    v35 = 0;
    v36 = -8192;
    do
    {
      v22 = v20[2];
      if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
        sub_BD60C0(v20);
      v20 += 3;
    }
    while ( v21 != v20 );
    if ( v36 != -4096 && v36 != 0 && v36 != -8192 )
      sub_BD60C0(&v34);
    if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
      sub_BD60C0(&v29);
    v11 = *(unsigned int *)(a1 + 88);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 24 * v11, 8);
  v12 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD **)(a1 + 40);
    v14 = &v13[5 * v12];
    while ( 1 )
    {
      while ( *v13 == -4096 )
      {
        if ( v13[1] != -4096 )
          goto LABEL_24;
        v13 += 5;
        if ( v14 == v13 )
        {
LABEL_31:
          v12 = *(unsigned int *)(a1 + 56);
          return sub_C7D6A0(*(_QWORD *)(a1 + 40), 40 * v12, 8);
        }
      }
      if ( *v13 != -8192 || v13[1] != -8192 )
      {
LABEL_24:
        v15 = v13[4];
        if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
          sub_BD60C0(v13 + 2);
      }
      v13 += 5;
      if ( v14 == v13 )
        goto LABEL_31;
    }
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 40), 40 * v12, 8);
}
