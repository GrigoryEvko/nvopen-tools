// Function: sub_2925AD0
// Address: 0x2925ad0
//
void __fastcall sub_2925AD0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v8; // rdi
  unsigned int v9; // ebx
  bool v10; // al
  __int64 v11; // rax
  _QWORD *v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  char v17; // cl
  __int64 v18; // rdi
  int v19; // esi
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r9
  unsigned __int64 *v23; // r14
  __int64 v24; // rbx
  unsigned int v25; // r15d
  __int64 *v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  unsigned int v32; // eax
  int v33; // edx
  unsigned int v34; // edi
  unsigned __int8 *v35; // rax
  int v36; // eax
  int v37; // r10d
  __int64 v38; // rsi
  int v39; // edx
  unsigned int v40; // eax
  __int64 v41; // rcx
  __int64 v42; // rsi
  int v43; // edx
  unsigned int v44; // eax
  __int64 v45; // rcx
  int v46; // r9d
  __int64 *v47; // rdi
  int v48; // edx
  int v49; // edx
  int v50; // r9d

  if ( !*(_QWORD *)(a2 + 16) )
  {
    sub_2916B30(a1, a2, a3, a4, (__int64)a5, a6);
    return;
  }
  if ( *(_BYTE *)a2 != 84 )
    goto LABEL_3;
  v15 = *(_QWORD *)(a2 + 40);
  v16 = sub_AA5190(v15);
  if ( v16 && v16 == v15 + 48 )
    goto LABEL_17;
  if ( *(_BYTE *)a2 == 84 )
  {
    v11 = sub_B48DC0(a2);
    if ( v11 )
      goto LABEL_7;
  }
  else
  {
LABEL_3:
    v8 = *(_QWORD *)(a2 - 96);
    if ( *(_BYTE *)v8 == 17 )
    {
      v9 = *(_DWORD *)(v8 + 32);
      if ( v9 <= 0x40 )
        v10 = *(_QWORD *)(v8 + 24) == 0;
      else
        v10 = v9 == (unsigned int)sub_C444A0(v8 + 24);
      v11 = *(_QWORD *)(a2 + 32LL * ((v10 + 1) & 3) - 96);
      if ( v11 )
      {
LABEL_7:
        v12 = *(_QWORD **)(a1 + 336);
        if ( *v12 == v11 )
        {
          sub_3109010(a1, a2);
        }
        else
        {
          v13 = *(_QWORD *)(a1 + 376);
          if ( !*(_BYTE *)v13 )
          {
            v14 = *(unsigned int *)(v13 + 400);
            if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 404) )
            {
              sub_C8D5F0(v13 + 392, (const void *)(v13 + 408), v14 + 1, 8u, (__int64)a5, a6);
              v14 = *(unsigned int *)(v13 + 400);
            }
            *(_QWORD *)(*(_QWORD *)(v13 + 392) + 8 * v14) = v12;
            ++*(_DWORD *)(v13 + 400);
          }
        }
        return;
      }
    }
    else
    {
      v11 = *(_QWORD *)(a2 - 64);
      if ( v11 )
      {
        v30 = *(_QWORD *)(a2 - 32);
        if ( v30 )
        {
          if ( v11 == v30 )
            goto LABEL_7;
        }
      }
    }
  }
  if ( !*(_BYTE *)(a1 + 344) )
  {
LABEL_17:
    *(_QWORD *)(a1 + 8) = a2;
    return;
  }
  v17 = *(_BYTE *)(a1 + 472) & 1;
  if ( v17 )
  {
    v18 = a1 + 480;
    v19 = 3;
  }
  else
  {
    v31 = *(_DWORD *)(a1 + 488);
    v18 = *(_QWORD *)(a1 + 480);
    if ( !v31 )
    {
      v32 = *(_DWORD *)(a1 + 472);
      ++*(_QWORD *)(a1 + 464);
      a5 = 0;
      v33 = (v32 >> 1) + 1;
LABEL_41:
      v34 = 3 * v31;
      goto LABEL_42;
    }
    v19 = v31 - 1;
  }
  v20 = v19 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v21 = (__int64 *)(v18 + 16LL * v20);
  v22 = *v21;
  if ( a2 == *v21 )
  {
LABEL_22:
    v23 = (unsigned __int64 *)(v21 + 1);
    if ( v21[1] )
      goto LABEL_23;
    goto LABEL_47;
  }
  v37 = 1;
  a5 = 0;
  while ( v22 != -4096 )
  {
    if ( v22 == -8192 && !a5 )
      a5 = v21;
    v20 = v19 & (v37 + v20);
    v21 = (__int64 *)(v18 + 16LL * v20);
    v22 = *v21;
    if ( a2 == *v21 )
      goto LABEL_22;
    ++v37;
  }
  v34 = 12;
  v31 = 4;
  if ( !a5 )
    a5 = v21;
  v32 = *(_DWORD *)(a1 + 472);
  ++*(_QWORD *)(a1 + 464);
  v33 = (v32 >> 1) + 1;
  if ( !v17 )
  {
    v31 = *(_DWORD *)(a1 + 488);
    goto LABEL_41;
  }
LABEL_42:
  if ( 4 * v33 >= v34 )
  {
    sub_29256B0(a1 + 464, 2 * v31);
    if ( (*(_BYTE *)(a1 + 472) & 1) != 0 )
    {
      v38 = a1 + 480;
      v39 = 3;
    }
    else
    {
      v48 = *(_DWORD *)(a1 + 488);
      v38 = *(_QWORD *)(a1 + 480);
      if ( !v48 )
        goto LABEL_89;
      v39 = v48 - 1;
    }
    v40 = v39 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    a5 = (__int64 *)(v38 + 16LL * v40);
    v41 = *a5;
    if ( a2 != *a5 )
    {
      v50 = 1;
      v47 = 0;
      while ( v41 != -4096 )
      {
        if ( v41 == -8192 && !v47 )
          v47 = a5;
        v40 = v39 & (v50 + v40);
        a5 = (__int64 *)(v38 + 16LL * v40);
        v41 = *a5;
        if ( a2 == *a5 )
          goto LABEL_60;
        ++v50;
      }
      goto LABEL_66;
    }
LABEL_60:
    v32 = *(_DWORD *)(a1 + 472);
    goto LABEL_44;
  }
  if ( v31 - *(_DWORD *)(a1 + 476) - v33 <= v31 >> 3 )
  {
    sub_29256B0(a1 + 464, v31);
    if ( (*(_BYTE *)(a1 + 472) & 1) != 0 )
    {
      v42 = a1 + 480;
      v43 = 3;
      goto LABEL_63;
    }
    v49 = *(_DWORD *)(a1 + 488);
    v42 = *(_QWORD *)(a1 + 480);
    if ( v49 )
    {
      v43 = v49 - 1;
LABEL_63:
      v44 = v43 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      a5 = (__int64 *)(v42 + 16LL * v44);
      v45 = *a5;
      if ( a2 != *a5 )
      {
        v46 = 1;
        v47 = 0;
        while ( v45 != -4096 )
        {
          if ( !v47 && v45 == -8192 )
            v47 = a5;
          v44 = v43 & (v46 + v44);
          a5 = (__int64 *)(v42 + 16LL * v44);
          v45 = *a5;
          if ( a2 == *a5 )
            goto LABEL_60;
          ++v46;
        }
LABEL_66:
        if ( v47 )
          a5 = v47;
        goto LABEL_60;
      }
      goto LABEL_60;
    }
LABEL_89:
    *(_DWORD *)(a1 + 472) = (2 * (*(_DWORD *)(a1 + 472) >> 1) + 2) | *(_DWORD *)(a1 + 472) & 1;
    BUG();
  }
LABEL_44:
  *(_DWORD *)(a1 + 472) = (2 * (v32 >> 1) + 2) | v32 & 1;
  if ( *a5 != -4096 )
    --*(_DWORD *)(a1 + 476);
  *a5 = a2;
  v23 = (unsigned __int64 *)(a5 + 1);
  a5[1] = 0;
LABEL_47:
  v35 = sub_291A1B0(a1, a2, v23);
  if ( v35 )
  {
    *(_QWORD *)(a1 + 8) = v35;
    return;
  }
LABEL_23:
  v24 = *(_QWORD *)(a1 + 376);
  if ( *(_BYTE *)v24 )
    return;
  v25 = *(_DWORD *)(a1 + 360);
  v26 = (__int64 *)(a1 + 352);
  if ( v25 <= 0x40 )
  {
    v27 = *(_QWORD *)(a1 + 352);
    goto LABEL_26;
  }
  v36 = sub_C444A0(a1 + 352);
  v26 = (__int64 *)(a1 + 352);
  if ( v25 - v36 <= 0x40 )
  {
    v27 = **(_QWORD **)(a1 + 352);
LABEL_26:
    if ( *(_QWORD *)(a1 + 368) > v27 )
    {
      sub_2916EE0(a1, a2, v26, *v23, 0);
      return;
    }
  }
  v28 = *(unsigned int *)(v24 + 400);
  v29 = *(_QWORD *)(a1 + 336);
  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(v24 + 404) )
  {
    sub_C8D5F0(v24 + 392, (const void *)(v24 + 408), v28 + 1, 8u, (__int64)a5, v22);
    v28 = *(unsigned int *)(v24 + 400);
  }
  *(_QWORD *)(*(_QWORD *)(v24 + 392) + 8 * v28) = v29;
  ++*(_DWORD *)(v24 + 400);
}
