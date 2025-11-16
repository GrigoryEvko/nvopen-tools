// Function: sub_1F09100
// Address: 0x1f09100
//
__int64 __fastcall sub_1F09100(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // esi
  int v5; // r14d
  __int64 v6; // r10
  __int64 v7; // r15
  int v8; // r13d
  unsigned __int64 v9; // r11
  unsigned int v10; // r8d
  bool v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  int v15; // ecx
  int v16; // ecx
  __int64 v17; // r9
  int v18; // r11d
  __int64 v19; // r10
  unsigned int v20; // edi
  unsigned __int64 v21; // r12
  __int64 v22; // r8
  unsigned __int64 v23; // r8
  unsigned int v24; // edi
  unsigned int v25; // r8d
  int v26; // ecx
  __int64 *v27; // rsi
  _QWORD *v28; // r13
  _QWORD *v29; // rcx
  _QWORD *v30; // rdx
  _QWORD *v31; // rdi
  _QWORD *v32; // rdi
  _QWORD *v33; // r12
  _QWORD *v34; // rdi
  __int64 v35; // r12
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 result; // rax
  int v39; // eax
  int v40; // eax
  int v41; // eax
  __int64 v42; // rdi
  int v43; // r9d
  unsigned int v44; // r14d
  __int64 v45; // r8
  unsigned __int64 v46; // r10
  __int64 v47; // rsi
  unsigned __int64 v48; // rsi
  unsigned int v49; // r14d
  __int64 v50; // [rsp+0h] [rbp-80h]
  __int64 v51; // [rsp+0h] [rbp-80h]
  _QWORD v53[4]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v54; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v55; // [rsp+38h] [rbp-48h] BYREF
  _QWORD *v56; // [rsp+40h] [rbp-40h]
  __int64 v57; // [rsp+48h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v5 = 37 * a3;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = 1;
  v9 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = (v4 - 1) & (37 * a3);
  v11 = !((a3 >> 2) & 1);
  while ( 1 )
  {
    v12 = v6 + 16LL * v10;
    v13 = *(_QWORD *)v12;
    if ( v11 != !((*(__int64 *)v12 >> 2) & 1) )
      break;
    v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
    if ( ((a3 >> 2) & 1) == 0 )
    {
      if ( v14 == v9 )
        goto LABEL_56;
      goto LABEL_8;
    }
    if ( v14 == v9 )
    {
LABEL_56:
      v35 = *(unsigned int *)(v12 + 8);
      goto LABEL_41;
    }
LABEL_25:
    v25 = v8 + v10;
    ++v8;
    v10 = (v4 - 1) & v25;
  }
  if ( ((*(__int64 *)v12 >> 2) & 1) != 0 )
    goto LABEL_25;
  v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_8:
  if ( v14 != -8 )
  {
    if ( v14 == -16 && !v7 )
      v7 = v6 + 16LL * v10;
    goto LABEL_25;
  }
  if ( !v7 )
    v7 = v6 + 16LL * v10;
  v39 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v26 = v39 + 1;
  if ( 4 * (v39 + 1) < 3 * v4 )
  {
    if ( v4 - *(_DWORD *)(a1 + 20) - v26 > v4 >> 3 )
      goto LABEL_29;
    v51 = a3;
    v7 = 0;
    sub_1F08EC0(a1, v4);
    v40 = *(_DWORD *)(a1 + 24);
    a3 = v51;
    if ( !v40 )
      goto LABEL_28;
    v41 = v40 - 1;
    v43 = 1;
    v44 = v41 & v5;
    v45 = 0;
    v46 = v51 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 2 )
    {
      v42 = *(_QWORD *)(a1 + 8);
      v7 = v42 + 16LL * v44;
      v47 = *(_QWORD *)v7;
      if ( v11 == !((*(__int64 *)v7 >> 2) & 1) )
      {
        v48 = v47 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v11 )
        {
          if ( v48 == v46 )
            goto LABEL_28;
LABEL_51:
          if ( v48 == -8 )
          {
            if ( v45 )
              v7 = v45;
            goto LABEL_28;
          }
          if ( v48 == -16 && !v45 )
            v45 = v42 + 16LL * v44;
        }
        else if ( v48 == v46 )
        {
          goto LABEL_28;
        }
      }
      else if ( ((*(__int64 *)v7 >> 2) & 1) == 0 )
      {
        v48 = v47 & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_51;
      }
      v49 = v43 + v44;
      ++v43;
      v44 = v41 & v49;
      continue;
    }
  }
LABEL_14:
  v50 = a3;
  v7 = 0;
  sub_1F08EC0(a1, 2 * v4);
  v15 = *(_DWORD *)(a1 + 24);
  a3 = v50;
  if ( !v15 )
    goto LABEL_28;
  v16 = v15 - 1;
  v17 = *(_QWORD *)(a1 + 8);
  v18 = 1;
  v19 = 0;
  v20 = v16 & (37 * v50);
  v21 = v50 & 0xFFFFFFFFFFFFFFF8LL;
  while ( 2 )
  {
    v7 = v17 + 16LL * v20;
    v22 = *(_QWORD *)v7;
    if ( !((v50 >> 2) & 1) != !((*(__int64 *)v7 >> 2) & 1) )
    {
      if ( ((*(__int64 *)v7 >> 2) & 1) == 0 )
      {
        v23 = v22 & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_19;
      }
LABEL_23:
      v24 = v18 + v20;
      ++v18;
      v20 = v16 & v24;
      continue;
    }
    break;
  }
  v23 = v22 & 0xFFFFFFFFFFFFFFF8LL;
  if ( ((v50 >> 2) & 1) != 0 )
  {
    if ( v21 == v23 )
      goto LABEL_28;
    goto LABEL_23;
  }
  if ( v21 == v23 )
    goto LABEL_28;
LABEL_19:
  if ( v23 != -8 )
  {
    if ( v23 == -16 && !v19 )
      v19 = v17 + 16LL * v20;
    goto LABEL_23;
  }
  if ( v19 )
    v7 = v19;
LABEL_28:
  v26 = *(_DWORD *)(a1 + 16) + 1;
LABEL_29:
  *(_DWORD *)(a1 + 16) = v26;
  if ( (*(_QWORD *)v7 & 4) != 0 || (*(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v7 = a3;
  *(_DWORD *)(v7 + 8) = 0;
  v27 = *(__int64 **)(a1 + 40);
  v53[1] = v53;
  v53[0] = v53;
  v53[2] = 0;
  v54 = a3;
  v57 = 0;
  v56 = &v55;
  v55 = &v55;
  if ( v27 == *(__int64 **)(a1 + 48) )
  {
    sub_1F08390((__int64 *)(a1 + 32), v27, &v54);
    v28 = v55;
  }
  else
  {
    v28 = &v55;
    if ( v27 )
    {
      *v27 = a3;
      v29 = v55;
      v30 = v27 + 1;
      v27[1] = (__int64)v55;
      v31 = v56;
      v27[2] = (__int64)v56;
      v27[3] = v57;
      if ( v29 == &v55 )
      {
        v27[1] = (__int64)v30;
        v28 = v55;
        v27[2] = (__int64)v30;
        v27 = *(__int64 **)(a1 + 40);
      }
      else
      {
        *v31 = v30;
        *(_QWORD *)(v27[1] + 8) = v30;
        v27 = *(__int64 **)(a1 + 40);
        v56 = &v55;
        v55 = &v55;
        v57 = 0;
      }
    }
    *(_QWORD *)(a1 + 40) = v27 + 4;
  }
  while ( v28 != &v55 )
  {
    v32 = v28;
    v28 = (_QWORD *)*v28;
    j_j___libc_free_0(v32, 24);
  }
  v33 = (_QWORD *)v53[0];
  while ( v33 != v53 )
  {
    v34 = v33;
    v33 = (_QWORD *)*v33;
    j_j___libc_free_0(v34, 24);
  }
  v35 = (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 5) - 1;
  *(_DWORD *)(v7 + 8) = v35;
LABEL_41:
  v36 = *(_QWORD *)(a1 + 32) + 32 * v35;
  v37 = sub_22077B0(24);
  *(_QWORD *)(v37 + 16) = a2;
  result = sub_2208C80(v37, v36 + 8);
  ++*(_QWORD *)(v36 + 24);
  ++*(_DWORD *)(a1 + 56);
  return result;
}
