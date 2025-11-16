// Function: sub_CFEC60
// Address: 0xcfec60
//
char __fastcall sub_CFEC60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  unsigned __int64 *v5; // r15
  bool v6; // di
  __int64 v7; // r9
  __int64 v8; // rax
  unsigned int v9; // edx
  __int64 v10; // r14
  __int64 v11; // rcx
  void *v12; // rax
  unsigned __int64 v13; // r13
  __int64 v14; // r8
  __int64 v15; // r12
  unsigned __int64 v16; // rbx
  __int64 v17; // r10
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // r9
  __int64 v20; // rdi
  unsigned __int64 v21; // r14
  __int64 v22; // rdx
  __int64 v23; // rcx
  _QWORD *v24; // rax
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 v29; // r15
  __int64 v30; // rdx
  _QWORD *v31; // r14
  _QWORD *v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  bool v36; // zf
  unsigned __int64 v37; // r10
  __int64 v38; // rax
  __int64 v39; // rdx
  bool v40; // al
  int v41; // r8d
  int v42; // edi
  __int64 v44; // [rsp+0h] [rbp-B0h]
  __int64 v45; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  bool v48; // [rsp+17h] [rbp-99h]
  _QWORD v49[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v50; // [rsp+38h] [rbp-78h]
  __int64 v51; // [rsp+40h] [rbp-70h]
  void *v52; // [rsp+50h] [rbp-60h]
  __int64 v53; // [rsp+58h] [rbp-58h] BYREF
  __int64 v54; // [rsp+60h] [rbp-50h]
  __int64 v55; // [rsp+68h] [rbp-48h]
  __int64 v56; // [rsp+70h] [rbp-40h]

  v3 = a2;
  v4 = a1;
  v55 = a2;
  v5 = sub_CFD720(a1, a3);
  v53 = 2;
  v54 = 0;
  v48 = v55 != -8192 && v55 != -4096 && v55 != 0;
  if ( v48 )
  {
    sub_BD73F0((__int64)&v53);
    a2 = v55;
    v6 = v55 != -8192 && v55 != 0 && v55 != -4096;
  }
  else
  {
    v6 = 0;
  }
  v56 = 0;
  v7 = *(_QWORD *)(v4 + 168);
  v52 = &unk_49DDAE8;
  v8 = *(unsigned int *)(v4 + 184);
  if ( !(_DWORD)v8 )
  {
LABEL_67:
    v10 = v7 + 88 * v8;
    v12 = &unk_49DB368;
    v52 = &unk_49DB368;
    if ( !v6 )
      return (char)v12;
    goto LABEL_68;
  }
  v9 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = v7 + 88LL * v9;
  v11 = *(_QWORD *)(v10 + 24);
  if ( v11 != a2 )
  {
    v41 = 1;
    while ( v11 != -4096 )
    {
      v9 = (v8 - 1) & (v9 + v41);
      v10 = v7 + 88LL * v9;
      v11 = *(_QWORD *)(v10 + 24);
      if ( v11 == a2 )
        goto LABEL_5;
      ++v41;
    }
    goto LABEL_67;
  }
LABEL_5:
  v52 = &unk_49DB368;
  if ( v6 )
  {
LABEL_68:
    sub_BD60C0(&v53);
    v8 = *(unsigned int *)(v4 + 184);
    v7 = *(_QWORD *)(v4 + 168);
  }
  v12 = (void *)(v7 + 88 * v8);
  if ( (void *)v10 == v12 )
    return (char)v12;
  v13 = *(_QWORD *)(v10 + 40);
  if ( v13 + 32LL * *(unsigned int *)(v10 + 48) == v13 )
    goto LABEL_19;
  v14 = v3;
  v15 = v4;
  v16 = v13 + 32LL * *(unsigned int *)(v10 + 48);
  do
  {
    v17 = *((unsigned int *)v5 + 2);
    v18 = *v5;
    v19 = v13;
    LODWORD(v20) = *((_DWORD *)v5 + 2);
    v21 = *v5 + 32 * v17;
    v22 = (32 * v17) >> 5;
    v23 = (32 * v17) >> 7;
    if ( v23 )
    {
      v22 = *(_QWORD *)(v13 + 16);
      v24 = (_QWORD *)*v5;
      v23 = v18 + (v23 << 7);
      while ( v24[2] != v22 )
      {
        if ( v22 == v24[6] )
        {
          v24 += 4;
          break;
        }
        if ( v22 == v24[10] )
        {
          v24 += 8;
          break;
        }
        if ( v22 == v24[14] )
        {
          v24 += 12;
          break;
        }
        v24 += 16;
        if ( (_QWORD *)v23 == v24 )
        {
          v22 = (__int64)(v21 - (_QWORD)v24) >> 5;
          goto LABEL_38;
        }
      }
LABEL_16:
      if ( (_QWORD *)v21 != v24 )
        goto LABEL_17;
LABEL_41:
      v37 = v17 + 1;
      if ( v37 > *((unsigned int *)v5 + 3) )
        goto LABEL_62;
      goto LABEL_42;
    }
    v24 = (_QWORD *)*v5;
LABEL_38:
    switch ( v22 )
    {
      case 2LL:
        v22 = *(_QWORD *)(v13 + 16);
        break;
      case 3LL:
        v22 = *(_QWORD *)(v13 + 16);
        if ( v24[2] == v22 )
          goto LABEL_16;
        v24 += 4;
        break;
      case 1LL:
        v22 = *(_QWORD *)(v13 + 16);
        goto LABEL_60;
      default:
        goto LABEL_41;
    }
    if ( v24[2] == v22 )
      goto LABEL_16;
    v24 += 4;
LABEL_60:
    if ( v24[2] == v22 )
      goto LABEL_16;
    v37 = v17 + 1;
    if ( v37 > *((unsigned int *)v5 + 3) )
    {
LABEL_62:
      if ( v18 > v13 || v21 <= v13 )
      {
        v45 = v14;
        sub_CFC2E0((__int64)v5, v37, v22, v23, v14, v13);
        v14 = v45;
        v19 = v13;
        v20 = *((unsigned int *)v5 + 2);
        v21 = *v5 + 32 * v20;
      }
      else
      {
        v47 = v14;
        sub_CFC2E0((__int64)v5, v37, v22, v23, v14, v13);
        v14 = v47;
        v19 = v13 - v18 + *v5;
        v20 = *((unsigned int *)v5 + 2);
        v21 = *v5 + 32 * v20;
      }
    }
LABEL_42:
    if ( v21 )
    {
      *(_QWORD *)v21 = 4;
      v38 = *(_QWORD *)(v19 + 16);
      *(_QWORD *)(v21 + 8) = 0;
      *(_QWORD *)(v21 + 16) = v38;
      if ( v38 != -4096 && v38 != 0 && v38 != -8192 )
      {
        v44 = v14;
        v46 = v19;
        sub_BD6050((unsigned __int64 *)v21, *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL);
        v14 = v44;
        v19 = v46;
      }
      *(_DWORD *)(v21 + 24) = *(_DWORD *)(v19 + 24);
      LODWORD(v20) = *((_DWORD *)v5 + 2);
    }
    *((_DWORD *)v5 + 2) = v20 + 1;
LABEL_17:
    v13 += 32LL;
  }
  while ( v16 != v13 );
  v4 = v15;
  v3 = v14;
LABEL_19:
  v49[1] = 0;
  v49[0] = 2;
  v50 = v3;
  if ( v48 )
  {
    sub_BD73F0((__int64)v49);
    v3 = v50;
  }
  v51 = 0;
  v25 = *(_DWORD *)(v4 + 184);
  if ( v25 )
  {
    v26 = v25 - 1;
    v27 = *(_QWORD *)(v4 + 168);
    v28 = (v25 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v29 = v27 + 88LL * v28;
    v30 = *(_QWORD *)(v29 + 24);
    if ( v30 == v3 )
    {
LABEL_23:
      v31 = *(_QWORD **)(v29 + 40);
      v32 = &v31[4 * *(unsigned int *)(v29 + 48)];
      if ( v31 != v32 )
      {
        do
        {
          v33 = *(v32 - 2);
          v32 -= 4;
          if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
            sub_BD60C0(v32);
        }
        while ( v31 != v32 );
        v32 = *(_QWORD **)(v29 + 40);
      }
      if ( v32 != (_QWORD *)(v29 + 56) )
        _libc_free(v32, v27);
      v53 = 2;
      v54 = 0;
      v55 = -8192;
      v52 = &unk_49DDAE8;
      v56 = 0;
      v34 = *(_QWORD *)(v29 + 24);
      if ( v34 == -8192 )
      {
        *(_QWORD *)(v29 + 32) = 0;
      }
      else
      {
        if ( !v34 || v34 == -4096 )
        {
          *(_QWORD *)(v29 + 24) = -8192;
          goto LABEL_52;
        }
        sub_BD60C0((_QWORD *)(v29 + 8));
        v35 = v55;
        v36 = v55 == -4096;
        *(_QWORD *)(v29 + 24) = v55;
        if ( v35 == 0 || v36 || v35 == -8192 )
        {
          *(_QWORD *)(v29 + 32) = v56;
        }
        else
        {
          sub_BD6050((unsigned __int64 *)(v29 + 8), v53 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_52:
          v39 = v55;
          v40 = v55 != -4096;
          v36 = v55 == 0;
          *(_QWORD *)(v29 + 32) = v56;
          v52 = &unk_49DB368;
          if ( v39 != -8192 && !v36 && v40 )
            sub_BD60C0(&v53);
        }
      }
      --*(_DWORD *)(v4 + 176);
      v3 = v50;
      ++*(_DWORD *)(v4 + 180);
    }
    else
    {
      v42 = 1;
      while ( v30 != -4096 )
      {
        v28 = v26 & (v42 + v28);
        v29 = v27 + 88LL * v28;
        v30 = *(_QWORD *)(v29 + 24);
        if ( v30 == v3 )
          goto LABEL_23;
        ++v42;
      }
    }
  }
  LOBYTE(v12) = v3 != -4096;
  if ( ((unsigned __int8)v12 & (v3 != 0)) != 0 && v3 != -8192 )
    LOBYTE(v12) = sub_BD60C0(v49);
  return (char)v12;
}
