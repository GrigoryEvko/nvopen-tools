// Function: sub_2F96FE0
// Address: 0x2f96fe0
//
__int64 __fastcall sub_2F96FE0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  char v9; // dl
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v16; // esi
  unsigned int v17; // eax
  __int64 v18; // r13
  int v19; // ecx
  unsigned int v20; // edi
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rsi
  int v25; // ecx
  __int64 v26; // rdx
  __int64 *v27; // rsi
  __int64 *v28; // rax
  _QWORD *v29; // rcx
  __int64 *v30; // rdi
  _QWORD *v31; // rdx
  _QWORD *v32; // r8
  __int64 v33; // r9
  _QWORD *v34; // rbx
  unsigned __int64 v35; // rdi
  _QWORD *v36; // rbx
  unsigned __int64 v37; // rdi
  __int64 v38; // rsi
  int v39; // eax
  unsigned int v40; // edx
  __int64 v41; // rcx
  unsigned __int64 v42; // rcx
  __int64 v43; // rdi
  __int64 v44; // rsi
  int v45; // eax
  unsigned int v46; // ecx
  __int64 v47; // rdx
  __int64 v48; // rdi
  int v49; // eax
  int v50; // eax
  __int64 v51; // [rsp+8h] [rbp-78h]
  _QWORD v52[4]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v53; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v54[9]; // [rsp+38h] [rbp-48h] BYREF

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( v9 )
  {
    v10 = a1 + 16;
    v11 = 3;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 24);
    v10 = *(_QWORD *)(a1 + 16);
    if ( !v16 )
    {
      v17 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v18 = 0;
      v19 = (v17 >> 1) + 1;
LABEL_9:
      v20 = 3 * v16;
      goto LABEL_10;
    }
    v11 = v16 - 1;
  }
  v12 = v11 & (37 * v8);
  v13 = v10 + 16LL * v12;
  a5 = *(_QWORD *)v13;
  if ( v8 == *(_QWORD *)v13 )
  {
LABEL_4:
    v14 = *(unsigned int *)(v13 + 8);
    return *(_QWORD *)(a1 + 80) + 32 * v14 + 8;
  }
  a6 = 1;
  v18 = 0;
  while ( a5 != -4096 )
  {
    if ( !v18 && a5 == -8192 )
      v18 = v13;
    v12 = v11 & (a6 + v12);
    v13 = v10 + 16LL * v12;
    a5 = *(_QWORD *)v13;
    if ( v8 == *(_QWORD *)v13 )
      goto LABEL_4;
    a6 = (unsigned int)(a6 + 1);
  }
  if ( !v18 )
    v18 = v13;
  v17 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v19 = (v17 >> 1) + 1;
  if ( !v9 )
  {
    v16 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v20 = 12;
  v16 = 4;
LABEL_10:
  if ( 4 * v19 >= v20 )
  {
    sub_2F96BB0(a1, 2 * v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v38 = a1 + 16;
      v39 = 3;
    }
    else
    {
      v49 = *(_DWORD *)(a1 + 24);
      v38 = *(_QWORD *)(a1 + 16);
      if ( !v49 )
        goto LABEL_69;
      v39 = v49 - 1;
    }
    v40 = v39 & (37 * v8);
    v18 = v38 + 16LL * v40;
    v41 = *(_QWORD *)v18;
    if ( v8 != *(_QWORD *)v18 )
    {
      a5 = 1;
      v48 = 0;
      while ( v41 != -4096 )
      {
        if ( !v48 && v41 == -8192 )
          v48 = v18;
        a6 = (unsigned int)(a5 + 1);
        v40 = v39 & (a5 + v40);
        v18 = v38 + 16LL * v40;
        v41 = *(_QWORD *)v18;
        if ( v8 == *(_QWORD *)v18 )
          goto LABEL_35;
        a5 = (unsigned int)a6;
      }
      goto LABEL_44;
    }
LABEL_35:
    v17 = *(_DWORD *)(a1 + 8);
    goto LABEL_12;
  }
  if ( v16 - *(_DWORD *)(a1 + 12) - v19 <= v16 >> 3 )
  {
    sub_2F96BB0(a1, v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v44 = a1 + 16;
      v45 = 3;
      goto LABEL_41;
    }
    v50 = *(_DWORD *)(a1 + 24);
    v44 = *(_QWORD *)(a1 + 16);
    if ( v50 )
    {
      v45 = v50 - 1;
LABEL_41:
      v46 = v45 & (37 * v8);
      v18 = v44 + 16LL * v46;
      v47 = *(_QWORD *)v18;
      if ( v8 != *(_QWORD *)v18 )
      {
        a5 = 1;
        v48 = 0;
        while ( v47 != -4096 )
        {
          if ( !v48 && v47 == -8192 )
            v48 = v18;
          a6 = (unsigned int)(a5 + 1);
          v46 = v45 & (a5 + v46);
          v18 = v44 + 16LL * v46;
          v47 = *(_QWORD *)v18;
          if ( v8 == *(_QWORD *)v18 )
            goto LABEL_35;
          a5 = (unsigned int)a6;
        }
LABEL_44:
        if ( v48 )
          v18 = v48;
        goto LABEL_35;
      }
      goto LABEL_35;
    }
LABEL_69:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 8) = (2 * (v17 >> 1) + 2) | v17 & 1;
  if ( *(_QWORD *)v18 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v18 = v8;
  *(_DWORD *)(v18 + 8) = 0;
  v21 = *a2;
  v22 = *(unsigned int *)(a1 + 92);
  v52[1] = v52;
  v53 = v21;
  v23 = *(unsigned int *)(a1 + 88);
  v52[0] = v52;
  v24 = v23 + 1;
  v52[2] = 0;
  v25 = v23;
  v54[2] = 0;
  v54[1] = v54;
  v54[0] = v54;
  if ( v23 + 1 > v22 )
  {
    v42 = *(_QWORD *)(a1 + 80);
    v43 = a1 + 80;
    if ( v42 > (unsigned __int64)&v53 || (v51 = *(_QWORD *)(a1 + 80), (unsigned __int64)&v53 >= v42 + 32 * v23) )
    {
      sub_2F96A60(v43, v24, v22, v42, a5, a6);
      v23 = *(unsigned int *)(a1 + 88);
      v26 = *(_QWORD *)(a1 + 80);
      v27 = &v53;
      v25 = *(_DWORD *)(a1 + 88);
    }
    else
    {
      sub_2F96A60(v43, v24, v22, v42, a5, a6);
      v26 = *(_QWORD *)(a1 + 80);
      v23 = *(unsigned int *)(a1 + 88);
      v27 = (_QWORD *)((char *)&v54[-1] + v26 - v51);
      v25 = *(_DWORD *)(a1 + 88);
    }
  }
  else
  {
    v26 = *(_QWORD *)(a1 + 80);
    v27 = &v53;
  }
  v28 = (__int64 *)(v26 + 32 * v23);
  if ( v28 )
  {
    v29 = v28 + 1;
    *v28 = *v27;
    v30 = (__int64 *)v27[1];
    v31 = v27 + 1;
    v32 = (_QWORD *)v27[2];
    v33 = v27[3];
    v28[1] = (__int64)v30;
    v28[2] = (__int64)v32;
    v28[3] = v33;
    if ( v27 + 1 == v30 )
    {
      v28[2] = (__int64)v29;
      v28[1] = (__int64)v29;
    }
    else
    {
      *v32 = v29;
      *(_QWORD *)(v28[1] + 8) = v29;
      v27[2] = (__int64)v31;
      v27[1] = (__int64)v31;
      v27[3] = 0;
    }
    v25 = *(_DWORD *)(a1 + 88);
  }
  v34 = (_QWORD *)v54[0];
  *(_DWORD *)(a1 + 88) = v25 + 1;
  while ( v34 != v54 )
  {
    v35 = (unsigned __int64)v34;
    v34 = (_QWORD *)*v34;
    j_j___libc_free_0(v35);
  }
  v36 = (_QWORD *)v52[0];
  while ( v36 != v52 )
  {
    v37 = (unsigned __int64)v36;
    v36 = (_QWORD *)*v36;
    j_j___libc_free_0(v37);
  }
  v14 = (unsigned int)(*(_DWORD *)(a1 + 88) - 1);
  *(_DWORD *)(v18 + 8) = v14;
  return *(_QWORD *)(a1 + 80) + 32 * v14 + 8;
}
