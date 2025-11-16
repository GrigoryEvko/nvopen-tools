// Function: sub_29D9270
// Address: 0x29d9270
//
__int64 __fastcall sub_29D9270(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  unsigned int v5; // esi
  int v6; // edx
  __int64 v7; // rax
  _QWORD *v8; // r12
  int v9; // ecx
  __int64 v10; // rdx
  unsigned __int64 *v11; // r13
  char v12; // r14
  __int64 v13; // rdi
  unsigned int v14; // edx
  __int64 v15; // rcx
  int v17; // ecx
  __int64 v18; // rdi
  _QWORD *v19; // r8
  int v20; // r9d
  unsigned int v21; // edx
  __int64 v22; // rsi
  int v23; // r10d
  _QWORD *v24; // r9
  int v25; // edi
  int v26; // edx
  int v27; // edx
  __int64 v28; // rdi
  int v29; // r9d
  unsigned int v30; // ecx
  __int64 v31; // rsi
  _QWORD v32[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+20h] [rbp-60h]
  void *v35; // [rsp+30h] [rbp-50h]
  unsigned __int64 v36; // [rsp+38h] [rbp-48h] BYREF
  __int64 v37; // [rsp+40h] [rbp-40h]
  __int64 v38; // [rsp+48h] [rbp-38h]
  __int64 v39; // [rsp+50h] [rbp-30h]
  __int64 v40; // [rsp+58h] [rbp-28h]

  v33 = a2;
  v3 = *(_QWORD *)(a1 + 80);
  v32[0] = 2;
  v32[1] = 0;
  if ( a2 == -4096 || a2 == 0 || a2 == -8192 )
  {
    v34 = a1;
    v36 = 2;
    v37 = 0;
    v38 = a2;
    v4 = a1;
  }
  else
  {
    sub_BD73F0((__int64)v32);
    v34 = a1;
    v37 = 0;
    v38 = v33;
    v36 = v32[0] & 6;
    if ( v33 == 0 || v33 == -4096 || v33 == -8192 )
    {
      v4 = a1;
    }
    else
    {
      sub_BD6050(&v36, v32[0] & 0xFFFFFFFFFFFFFFF8LL);
      v4 = v34;
    }
  }
  v5 = *(_DWORD *)(a1 + 24);
  v39 = v4;
  v35 = &unk_4A1FB50;
  v40 = v3;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
LABEL_9:
    sub_29D8E20(a1, 2 * v5);
    v6 = *(_DWORD *)(a1 + 24);
    if ( !v6 )
    {
LABEL_10:
      v7 = v38;
      v8 = 0;
LABEL_11:
      v9 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_12;
    }
    v7 = v38;
    v17 = v6 - 1;
    v18 = *(_QWORD *)(a1 + 8);
    v19 = 0;
    v20 = 1;
    v21 = (v6 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
    v8 = (_QWORD *)(v18 + 48LL * v21);
    v22 = v8[3];
    if ( v22 == v38 )
      goto LABEL_11;
    while ( v22 != -4096 )
    {
      if ( !v19 && v22 == -8192 )
        v19 = v8;
      v21 = v17 & (v20 + v21);
      v8 = (_QWORD *)(v18 + 48LL * v21);
      v22 = v8[3];
      if ( v38 == v22 )
        goto LABEL_11;
      ++v20;
    }
LABEL_35:
    if ( v19 )
      v8 = v19;
    goto LABEL_11;
  }
  v7 = v38;
  v13 = *(_QWORD *)(a1 + 8);
  v14 = (v5 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
  v8 = (_QWORD *)(v13 + 48LL * v14);
  v15 = v8[3];
  if ( v38 == v15 )
  {
LABEL_23:
    v12 = 0;
    goto LABEL_24;
  }
  v23 = 1;
  v24 = 0;
  while ( v15 != -4096 )
  {
    if ( !v24 && v15 == -8192 )
      v24 = v8;
    v14 = (v5 - 1) & (v23 + v14);
    v8 = (_QWORD *)(v13 + 48LL * v14);
    v15 = v8[3];
    if ( v38 == v15 )
      goto LABEL_23;
    ++v23;
  }
  v25 = *(_DWORD *)(a1 + 16);
  if ( v24 )
    v8 = v24;
  ++*(_QWORD *)a1;
  v9 = v25 + 1;
  if ( 4 * (v25 + 1) >= 3 * v5 )
    goto LABEL_9;
  if ( v5 - *(_DWORD *)(a1 + 20) - v9 <= v5 >> 3 )
  {
    sub_29D8E20(a1, v5);
    v26 = *(_DWORD *)(a1 + 24);
    if ( !v26 )
      goto LABEL_10;
    v7 = v38;
    v27 = v26 - 1;
    v28 = *(_QWORD *)(a1 + 8);
    v19 = 0;
    v29 = 1;
    v30 = v27 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
    v8 = (_QWORD *)(v28 + 48LL * v30);
    v31 = v8[3];
    if ( v38 == v31 )
      goto LABEL_11;
    while ( v31 != -4096 )
    {
      if ( v31 == -8192 && !v19 )
        v19 = v8;
      v30 = v27 & (v29 + v30);
      v8 = (_QWORD *)(v28 + 48LL * v30);
      v31 = v8[3];
      if ( v38 == v31 )
        goto LABEL_11;
      ++v29;
    }
    goto LABEL_35;
  }
LABEL_12:
  *(_DWORD *)(a1 + 16) = v9;
  if ( v8[3] == -4096 )
  {
    v11 = v8 + 1;
    if ( v7 != -4096 )
    {
LABEL_17:
      v8[3] = v7;
      if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
        sub_BD6050(v11, v36 & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v38;
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 20);
    v10 = v8[3];
    if ( v10 != v7 )
    {
      v11 = v8 + 1;
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
      {
        sub_BD60C0(v8 + 1);
        v7 = v38;
      }
      goto LABEL_17;
    }
  }
  v12 = 1;
  v8[4] = v39;
  v8[5] = v40;
LABEL_24:
  v35 = &unk_49DB368;
  if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
    sub_BD60C0(&v36);
  if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
    sub_BD60C0(v32);
  if ( v12 )
    ++*(_QWORD *)(a1 + 80);
  return v8[5];
}
