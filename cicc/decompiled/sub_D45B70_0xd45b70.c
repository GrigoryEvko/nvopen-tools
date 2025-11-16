// Function: sub_D45B70
// Address: 0xd45b70
//
__int64 __fastcall sub_D45B70(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  int v10; // eax
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r14
  int v14; // ecx
  __int64 v15; // rdx
  unsigned __int64 *v16; // r13
  char v17; // r13
  __int64 v18; // r15
  __int64 v19; // r15
  unsigned int v20; // edx
  __int64 v21; // rdi
  int v23; // ecx
  __int64 v24; // rdi
  __int64 v25; // r8
  int v26; // r9d
  unsigned int v27; // edx
  __int64 v28; // rsi
  int v29; // r10d
  __int64 v30; // r9
  int v31; // edx
  int v32; // edx
  __int64 v33; // rdi
  int v34; // r9d
  unsigned int v35; // ecx
  __int64 v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-98h]
  __int64 v38; // [rsp+8h] [rbp-98h]
  _QWORD v39[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v40; // [rsp+28h] [rbp-78h]
  __int64 v41; // [rsp+30h] [rbp-70h]
  void *v42; // [rsp+40h] [rbp-60h]
  unsigned __int64 v43; // [rsp+48h] [rbp-58h] BYREF
  __int64 v44; // [rsp+50h] [rbp-50h]
  __int64 v45; // [rsp+58h] [rbp-48h]
  __int64 v46; // [rsp+60h] [rbp-40h]
  int v47; // [rsp+68h] [rbp-38h]

  v6 = *a3;
  v39[1] = 0;
  v39[0] = 2;
  v40 = v6;
  if ( v6 == 0 || v6 == -4096 || v6 == -8192 )
  {
    v41 = a2;
    v43 = 2;
    v44 = 0;
    v45 = v6;
    v7 = a2;
  }
  else
  {
    sub_BD73F0((__int64)v39);
    v41 = a2;
    v44 = 0;
    v45 = v40;
    v43 = v39[0] & 6;
    if ( v40 == 0 || v40 == -4096 || v40 == -8192 )
    {
      v7 = a2;
    }
    else
    {
      sub_BD6050(&v43, v39[0] & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v41;
    }
  }
  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  v46 = v7;
  v10 = *((_DWORD *)a3 + 2);
  v42 = &unk_49DDFA0;
  v47 = v10;
  if ( !(_DWORD)v8 )
  {
    *(_QWORD *)a2 = v9 + 1;
LABEL_9:
    sub_D45720(a2, 2 * v8);
    v11 = *(_DWORD *)(a2 + 24);
    if ( !v11 )
    {
LABEL_10:
      v12 = v45;
      v13 = 0;
LABEL_11:
      v14 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_12;
    }
    v12 = v45;
    v23 = v11 - 1;
    v24 = *(_QWORD *)(a2 + 8);
    v25 = 0;
    v26 = 1;
    v27 = (v11 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
    v13 = v24 + 48LL * v27;
    v28 = *(_QWORD *)(v13 + 24);
    if ( v45 == v28 )
      goto LABEL_11;
    while ( v28 != -4096 )
    {
      if ( !v25 && v28 == -8192 )
        v25 = v13;
      v27 = v23 & (v26 + v27);
      v13 = v24 + 48LL * v27;
      v28 = *(_QWORD *)(v13 + 24);
      if ( v45 == v28 )
        goto LABEL_11;
      ++v26;
    }
LABEL_33:
    if ( v25 )
      v13 = v25;
    goto LABEL_11;
  }
  v12 = v45;
  v19 = *(_QWORD *)(a2 + 8);
  v20 = (v8 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
  v13 = v19 + 48LL * v20;
  v21 = *(_QWORD *)(v13 + 24);
  if ( v45 == v21 )
  {
LABEL_23:
    v17 = 0;
    v18 = 48 * v8 + v19;
    goto LABEL_24;
  }
  v29 = 1;
  v30 = 0;
  while ( v21 != -4096 )
  {
    if ( !v30 && v21 == -8192 )
      v30 = v13;
    v20 = (v8 - 1) & (v29 + v20);
    v13 = v19 + 48LL * v20;
    v21 = *(_QWORD *)(v13 + 24);
    if ( v45 == v21 )
      goto LABEL_23;
    ++v29;
  }
  if ( v30 )
    v13 = v30;
  *(_QWORD *)a2 = v9 + 1;
  v14 = *(_DWORD *)(a2 + 16) + 1;
  if ( 4 * v14 >= (unsigned int)(3 * v8) )
    goto LABEL_9;
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v14 <= (unsigned int)v8 >> 3 )
  {
    sub_D45720(a2, v8);
    v31 = *(_DWORD *)(a2 + 24);
    if ( !v31 )
      goto LABEL_10;
    v12 = v45;
    v32 = v31 - 1;
    v33 = *(_QWORD *)(a2 + 8);
    v25 = 0;
    v34 = 1;
    v35 = v32 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
    v13 = v33 + 48LL * v35;
    v36 = *(_QWORD *)(v13 + 24);
    if ( v36 == v45 )
      goto LABEL_11;
    while ( v36 != -4096 )
    {
      if ( !v25 && v36 == -8192 )
        v25 = v13;
      v35 = v32 & (v34 + v35);
      v13 = v33 + 48LL * v35;
      v36 = *(_QWORD *)(v13 + 24);
      if ( v45 == v36 )
        goto LABEL_11;
      ++v34;
    }
    goto LABEL_33;
  }
LABEL_12:
  *(_DWORD *)(a2 + 16) = v14;
  if ( *(_QWORD *)(v13 + 24) == -4096 )
  {
    v16 = (unsigned __int64 *)(v13 + 8);
    if ( v12 != -4096 )
    {
LABEL_17:
      *(_QWORD *)(v13 + 24) = v12;
      if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
        sub_BD6050(v16, v43 & 0xFFFFFFFFFFFFFFF8LL);
      v12 = v45;
      goto LABEL_21;
    }
    v15 = -4096;
  }
  else
  {
    --*(_DWORD *)(a2 + 20);
    v15 = *(_QWORD *)(v13 + 24);
    if ( v12 != v15 )
    {
      v16 = (unsigned __int64 *)(v13 + 8);
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
      {
        sub_BD60C0((_QWORD *)(v13 + 8));
        v12 = v45;
      }
      goto LABEL_17;
    }
  }
  v12 = v15;
LABEL_21:
  v17 = 1;
  *(_QWORD *)(v13 + 32) = v46;
  *(_DWORD *)(v13 + 40) = v47;
  v9 = *(_QWORD *)a2;
  v18 = *(_QWORD *)(a2 + 8) + 48LL * *(unsigned int *)(a2 + 24);
LABEL_24:
  v42 = &unk_49DB368;
  if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
  {
    v37 = v9;
    sub_BD60C0(&v43);
    v9 = v37;
  }
  if ( v40 != -4096 && v40 != 0 && v40 != -8192 )
  {
    v38 = v9;
    sub_BD60C0(v39);
    v9 = v38;
  }
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v13;
  *(_QWORD *)(a1 + 24) = v18;
  *(_BYTE *)(a1 + 32) = v17;
  *(_QWORD *)(a1 + 8) = v9;
  return a1;
}
