// Function: sub_F81E90
// Address: 0xf81e90
//
__int64 __fastcall sub_F81E90(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r12
  int v4; // r15d
  unsigned int v5; // ebx
  __int64 v6; // r13
  unsigned int v7; // r15d
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rbx
  char v20; // al
  __int64 v21; // rax
  int v22; // [rsp+14h] [rbp-9Ch]
  __int64 v23; // [rsp+18h] [rbp-98h]
  _QWORD v24[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v25; // [rsp+38h] [rbp-78h]
  char v26; // [rsp+40h] [rbp-70h]
  void *v27; // [rsp+50h] [rbp-60h]
  __int64 v28; // [rsp+58h] [rbp-58h] BYREF
  __int64 v29; // [rsp+60h] [rbp-50h]
  __int64 v30; // [rsp+68h] [rbp-48h]
  unsigned __int8 v31; // [rsp+70h] [rbp-40h]

  result = *(unsigned int *)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v22 = *(_DWORD *)(a1 + 16);
  if ( !(_DWORD)v3 )
  {
    if ( (_DWORD)result )
    {
      result = 0;
      goto LABEL_4;
    }
LABEL_48:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v15 = *(_QWORD *)(a1 + 8);
  v24[0] = 2;
  v24[1] = 0;
  v26 = 0;
  v16 = v15 + 48LL * (unsigned int)v3;
  v25 = -4096;
  v28 = 2;
  v29 = 0;
  v27 = &unk_49E51C0;
  v31 = 0;
  v30 = -8192;
  do
  {
    while ( *(_BYTE *)(v15 + 32) )
    {
      *(_QWORD *)(v15 + 24) = 0;
      v15 += 48;
      *(_QWORD *)(v15 - 48) = &unk_49DB368;
      if ( v16 == v15 )
        goto LABEL_26;
    }
    v17 = *(_QWORD *)(v15 + 24);
    *(_QWORD *)v15 = &unk_49DB368;
    if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
    {
      v23 = v15;
      sub_BD60C0((_QWORD *)(v15 + 8));
      v15 = v23;
    }
    v15 += 48;
  }
  while ( v16 != v15 );
LABEL_26:
  if ( !v31 )
  {
    v27 = &unk_49DB368;
    if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
      sub_BD60C0(&v28);
  }
  if ( !v26 && v25 != 0 && v25 != -4096 && v25 != -8192 )
    sub_BD60C0(v24);
  result = *(unsigned int *)(a1 + 24);
  if ( !v22 )
  {
    v18 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)result )
    {
      result = sub_C7D6A0(v18, 48LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_48;
  }
LABEL_4:
  v4 = 64;
  if ( v22 != 1 )
  {
    _BitScanReverse(&v5, v22 - 1);
    v4 = 1 << (33 - (v5 ^ 0x1F));
    if ( v4 < 64 )
      v4 = 64;
  }
  v6 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)result != v4 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 8), 48 * v3, 8);
    v7 = 4 * v4 / 3u;
    v8 = ((((((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
            | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
            | (v7 + 1)
            | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
          | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
          | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
          | (v7 + 1)
          | ((unsigned __int64)(v7 + 1) >> 1)) >> 16)
        | (((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
          | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
          | (v7 + 1)
          | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
        | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
        | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
        | (v7 + 1)
        | ((unsigned __int64)(v7 + 1) >> 1))
       + 1;
    *(_DWORD *)(a1 + 24) = v8;
    v9 = sub_C7D670(48 * v8, 8);
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v9;
    *(_QWORD *)(a1 + 8) = v9;
    v28 = 2;
    result = *(unsigned int *)(a1 + 24);
    v29 = 0;
    v27 = &unk_49E51C0;
    v31 = 0;
    v30 = -4096;
    v11 = v10 + 48 * result;
    if ( v10 == v11 )
      return result;
    do
    {
      if ( v10 )
      {
        v12 = v28;
        *(_QWORD *)(v10 + 16) = 0;
        *(_QWORD *)(v10 + 8) = v12 & 6;
        v13 = v30;
        v14 = v30 == -4096;
        *(_QWORD *)(v10 + 24) = v30;
        if ( v13 != 0 && !v14 && v13 != -8192 )
          sub_BD6050((unsigned __int64 *)(v10 + 8), v12 & 0xFFFFFFFFFFFFFFF8LL);
        result = v31;
        *(_QWORD *)v10 = &unk_49E51C0;
        *(_BYTE *)(v10 + 32) = result;
      }
      v10 += 48;
    }
    while ( v11 != v10 );
    if ( v31 )
      return result;
    result = v30;
    v27 = &unk_49DB368;
    if ( v30 == -4096 || v30 == 0 || v30 == -8192 )
      return result;
    return sub_BD60C0(&v28);
  }
  *(_QWORD *)(a1 + 16) = 0;
  v28 = 2;
  v19 = v6 + 48 * result;
  v31 = 0;
  v29 = 0;
  v27 = &unk_49E51C0;
  v30 = -4096;
  if ( v19 != v6 )
  {
    do
    {
      if ( v6 )
      {
        v20 = v28;
        *(_QWORD *)(v6 + 16) = 0;
        *(_QWORD *)(v6 + 8) = v20 & 6;
        v21 = v30;
        v14 = v30 == 0;
        *(_QWORD *)(v6 + 24) = v30;
        if ( v21 != -4096 && !v14 && v21 != -8192 )
          sub_BD6050((unsigned __int64 *)(v6 + 8), v28 & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)v6 = &unk_49E51C0;
        result = v31;
        *(_BYTE *)(v6 + 32) = v31;
      }
      v6 += 48;
    }
    while ( v19 != v6 );
    if ( !v31 )
    {
      result = v30;
      v27 = &unk_49DB368;
      if ( v30 != 0 && v30 != -8192 && v30 != -4096 )
        return sub_BD60C0(&v28);
    }
  }
  return result;
}
