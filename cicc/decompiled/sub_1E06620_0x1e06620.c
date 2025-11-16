// Function: sub_1E06620
// Address: 0x1e06620
//
void __fastcall sub_1E06620(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // r14
  _QWORD *v6; // rdx
  _QWORD *v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r12
  __int64 *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rdi
  unsigned int v18; // edx
  __int64 v19; // rdx
  _QWORD *v20; // rdx
  __int64 *v21; // r12
  __int64 v22; // r13
  __int64 v23; // rsi
  __int64 v24; // r15
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r14
  __int64 *v29; // rax
  __int64 v30; // r8
  __int64 v31; // rdi
  __int64 v32; // r8
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rax
  void *v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // r13
  __int64 *v41; // [rsp+8h] [rbp-A8h]
  __int64 v42; // [rsp+18h] [rbp-98h]
  __int64 *v43; // [rsp+18h] [rbp-98h]
  __int64 v44; // [rsp+18h] [rbp-98h]
  __int64 v45; // [rsp+18h] [rbp-98h]
  __int64 v46; // [rsp+28h] [rbp-88h]
  unsigned int v47; // [rsp+30h] [rbp-80h]
  unsigned __int64 v48; // [rsp+38h] [rbp-78h]
  __int64 v49; // [rsp+40h] [rbp-70h]
  char v50; // [rsp+4Fh] [rbp-61h]
  __int64 v51; // [rsp+50h] [rbp-60h]
  __int64 v52; // [rsp+50h] [rbp-60h]
  __int64 *v53; // [rsp+50h] [rbp-60h]
  __int64 *v54; // [rsp+58h] [rbp-58h]
  __int64 v55; // [rsp+58h] [rbp-58h]
  __int64 v56; // [rsp+68h] [rbp-48h] BYREF
  __int64 v57; // [rsp+70h] [rbp-40h] BYREF
  char *v58; // [rsp+78h] [rbp-38h] BYREF

  v1 = *(unsigned int *)(a1 + 240);
  if ( !(_DWORD)v1 )
    return;
  v2 = a1;
  if ( (unsigned int)v1 > 0x39 )
  {
    v39 = sub_22077B0(24);
    v40 = v39;
    if ( v39 )
      sub_1BFC1A0(v39, v1, 1u);
    v48 = v40;
    v3 = *(unsigned int *)(a1 + 240);
  }
  else
  {
    v3 = (unsigned int)v1;
    v48 = 2 * (~(-1LL << v1) | (v1 << 57)) + 1;
  }
  v4 = *(_QWORD *)(a1 + 232);
  v49 = a1 + 1016;
  v46 = v4 + 24 * v3;
  if ( v4 != v46 )
  {
    v47 = 0;
LABEL_25:
    v13 = *(_QWORD *)(v4 + 8);
    v14 = sub_1E05220(*(_QWORD *)(a1 + 1312), v13);
    v54 = *(__int64 **)(v13 + 72);
    v50 = v48 & 1;
    if ( v54 == *(__int64 **)(v13 + 64) )
      goto LABEL_24;
    v15 = *(__int64 **)(v13 + 64);
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = *v15;
        if ( *(_QWORD *)(v4 + 16) != *v15 )
          break;
LABEL_28:
        if ( v54 == ++v15 )
          goto LABEL_24;
      }
      v6 = *(_QWORD **)(a1 + 1032);
      v7 = *(_QWORD **)(a1 + 1024);
      if ( v6 == v7 )
      {
        v8 = &v7[*(unsigned int *)(a1 + 1044)];
        if ( v7 == v8 )
        {
          v20 = *(_QWORD **)(a1 + 1024);
        }
        else
        {
          do
          {
            if ( v9 == *v7 )
              break;
            ++v7;
          }
          while ( v8 != v7 );
          v20 = v8;
        }
        goto LABEL_37;
      }
      v51 = *v15;
      v8 = &v6[*(unsigned int *)(a1 + 1040)];
      v7 = sub_16CC9F0(v49, v9);
      v9 = v51;
      if ( v51 == *v7 )
        break;
      v10 = *(_QWORD *)(a1 + 1032);
      if ( v10 == *(_QWORD *)(a1 + 1024) )
      {
        v7 = (_QWORD *)(v10 + 8LL * *(unsigned int *)(a1 + 1044));
        v20 = v7;
LABEL_37:
        while ( v20 != v7 && *v7 >= 0xFFFFFFFFFFFFFFFELL )
          ++v7;
        goto LABEL_11;
      }
      v7 = (_QWORD *)(v10 + 8LL * *(unsigned int *)(a1 + 1040));
LABEL_11:
      if ( v7 != v8 )
        v9 = **(_QWORD **)(v9 + 64);
      v11 = *(_QWORD *)(a1 + 1312);
      v12 = sub_1E05220(v11, v9);
      if ( !v12 || v12 == v14 )
        goto LABEL_28;
      if ( !v14 )
        goto LABEL_22;
      if ( v14 == *(_QWORD *)(v12 + 8) )
        goto LABEL_28;
      if ( v12 == *(_QWORD *)(v14 + 8) || *(_DWORD *)(v14 + 16) >= *(_DWORD *)(v12 + 16) )
        goto LABEL_22;
      if ( *(_BYTE *)(v11 + 72) )
      {
        if ( *(_DWORD *)(v12 + 48) < *(_DWORD *)(v14 + 48) )
          goto LABEL_22;
LABEL_21:
        if ( *(_DWORD *)(v12 + 52) > *(_DWORD *)(v14 + 52) )
          goto LABEL_22;
        goto LABEL_28;
      }
      v18 = *(_DWORD *)(v11 + 76) + 1;
      *(_DWORD *)(v11 + 76) = v18;
      if ( v18 > 0x20 )
      {
        v52 = v12;
        sub_1E052A0(v11);
        v12 = v52;
        if ( *(_DWORD *)(v52 + 48) < *(_DWORD *)(v14 + 48) )
          goto LABEL_22;
        goto LABEL_21;
      }
      do
      {
        v19 = v12;
        v12 = *(_QWORD *)(v12 + 8);
      }
      while ( v12 && *(_DWORD *)(v14 + 16) <= *(_DWORD *)(v12 + 16) );
      if ( v14 != v19 )
      {
LABEL_22:
        if ( v50 )
        {
          v50 = 1;
          v48 = 2 * ((v48 >> 58 << 57) | (v48 >> 1) & ~((-1LL << (v48 >> 58)) | (1LL << v47))) + 1;
        }
        else
        {
          *(_QWORD *)(*(_QWORD *)v48 + 8LL * (v47 >> 6)) &= ~(1LL << v47);
        }
LABEL_24:
        ++v47;
        v4 += 24;
        if ( v4 != v46 )
          goto LABEL_25;
        v21 = *(__int64 **)(a1 + 232);
        v2 = a1;
        v53 = &v21[3 * *(unsigned int *)(a1 + 240)];
        if ( v21 == v53 )
          goto LABEL_69;
        v55 = a1;
        v22 = 0;
        while ( 1 )
        {
          v23 = *v21;
          v25 = *(_QWORD *)(v55 + 1312);
          v56 = v21[2];
          v24 = v56;
          v26 = sub_1E05220(v25, v23);
          *(_BYTE *)(v25 + 72) = 0;
          v27 = v26;
          sub_1E04AB0(&v57, v24, v26);
          v58 = (char *)v57;
          sub_1E06030(v27 + 24, &v58);
          v28 = v57;
          v57 = 0;
          v29 = sub_1E063B0(v25 + 24, &v56);
          v30 = v29[1];
          v29[1] = v28;
          if ( v30 )
          {
            v31 = *(_QWORD *)(v30 + 24);
            if ( v31 )
            {
              v41 = v29;
              v42 = v30;
              j_j___libc_free_0(v31, *(_QWORD *)(v30 + 40) - v31);
              v29 = v41;
              v30 = v42;
            }
            v43 = v29;
            j_j___libc_free_0(v30, 56);
            v28 = v43[1];
          }
          v32 = v57;
          if ( v57 )
          {
            v33 = *(_QWORD *)(v57 + 24);
            if ( v33 )
            {
              v44 = v57;
              j_j___libc_free_0(v33, *(_QWORD *)(v57 + 40) - v33);
              v32 = v44;
            }
            j_j___libc_free_0(v32, 56);
          }
          if ( v50 )
          {
            if ( ((((v48 >> 1) & ~(-1LL << (v48 >> 58))) >> v22) & 1) == 0 )
              goto LABEL_56;
          }
          else if ( ((*(_QWORD *)(*(_QWORD *)v48 + 8LL * ((unsigned int)v22 >> 6)) >> v22) & 1) == 0 )
          {
LABEL_56:
            ++v22;
            v21 += 3;
            if ( v53 == v21 )
              goto LABEL_68;
            continue;
          }
          v34 = v21[1];
          ++v22;
          v21 += 3;
          v45 = *(_QWORD *)(v55 + 1312);
          v35 = sub_1E05220(v45, v34);
          *(_BYTE *)(v45 + 72) = 0;
          sub_1E06060(v35, v28);
          if ( v53 == v21 )
          {
LABEL_68:
            v2 = v55;
            goto LABEL_69;
          }
        }
      }
      if ( v54 == ++v15 )
        goto LABEL_24;
    }
    v16 = *(_QWORD *)(a1 + 1032);
    if ( v16 == *(_QWORD *)(a1 + 1024) )
      v17 = *(unsigned int *)(a1 + 1044);
    else
      v17 = *(unsigned int *)(a1 + 1040);
    v20 = (_QWORD *)(v16 + 8 * v17);
    goto LABEL_37;
  }
  v50 = v48 & 1;
LABEL_69:
  ++*(_QWORD *)(v2 + 1016);
  v36 = *(void **)(v2 + 1032);
  if ( v36 == *(void **)(v2 + 1024) )
    goto LABEL_74;
  v37 = 4 * (*(_DWORD *)(v2 + 1044) - *(_DWORD *)(v2 + 1048));
  v38 = *(unsigned int *)(v2 + 1040);
  if ( v37 < 0x20 )
    v37 = 32;
  if ( v37 >= (unsigned int)v38 )
  {
    memset(v36, -1, 8 * v38);
LABEL_74:
    *(_QWORD *)(v2 + 1044) = 0;
    goto LABEL_75;
  }
  sub_16CC920(v49);
LABEL_75:
  *(_DWORD *)(v2 + 240) = 0;
  if ( !v50 )
  {
    if ( v48 )
    {
      _libc_free(*(_QWORD *)v48);
      j_j___libc_free_0(v48, 24);
    }
  }
}
