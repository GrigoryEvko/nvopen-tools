// Function: sub_390F280
// Address: 0x390f280
//
unsigned __int64 __fastcall sub_390F280(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rdx
  unsigned __int64 v12; // r13
  int v14; // edx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rsi
  __int64 v23; // rdx
  _QWORD *v24; // rcx
  unsigned int v25; // esi
  __int64 v26; // rdi
  int v27; // r15d
  __int64 v28; // rcx
  __int64 *v29; // r9
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // r11
  _QWORD *v33; // rax
  int v34; // r10d
  int v35; // eax
  int v36; // eax
  int v37; // edi
  int v38; // edi
  __int64 v39; // r10
  unsigned int v40; // edx
  __int64 v41; // r8
  int v42; // esi
  __int64 *v43; // rcx
  int v44; // edi
  int v45; // edi
  int v46; // edx
  __int64 *v47; // rsi
  __int64 v48; // r14
  __int64 v49; // r8
  __int64 v50; // rcx

  v9 = *(unsigned int *)(a1 + 152);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a1 + 136);
    a4 = ((_DWORD)v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = (__int64 *)(v10 + 16 * a4);
    a6 = *v11;
    if ( a2 == *v11 )
    {
LABEL_3:
      if ( v11 != (__int64 *)(v10 + 16 * v9) )
        return v11[1];
    }
    else
    {
      v14 = 1;
      while ( a6 != -8 )
      {
        v34 = v14 + 1;
        a4 = ((_DWORD)v9 - 1) & (unsigned int)(v14 + a4);
        v11 = (__int64 *)(v10 + 16LL * (unsigned int)a4);
        a6 = *v11;
        if ( a2 == *v11 )
          goto LABEL_3;
        v14 = v34;
      }
    }
  }
  v15 = 0;
  v16 = sub_390EBE0(a1, a2, a3, a4, a3, a6);
  v17 = *(__int64 **)v16;
  v18 = *(_QWORD *)v16 + 8LL * *(unsigned int *)(v16 + 8);
  if ( *(_QWORD *)v16 != v18 )
  {
    do
    {
      v19 = *v17++;
      v15 |= *(_QWORD *)(v19 + 48);
    }
    while ( v17 != (__int64 *)v18 );
  }
  v20 = *(_QWORD **)(a1 + 32);
  if ( v20 == *(_QWORD **)(a1 + 24) )
    v21 = *(unsigned int *)(a1 + 44);
  else
    v21 = *(unsigned int *)(a1 + 40);
  v22 = &v20[v21];
  if ( v20 == v22 )
    goto LABEL_15;
  while ( 1 )
  {
    v23 = *v20;
    v24 = v20;
    if ( *v20 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v22 == ++v20 )
      goto LABEL_15;
  }
  if ( v20 == v22 )
  {
LABEL_15:
    v12 = 0;
  }
  else
  {
    v12 = 0;
    do
    {
      if ( (*(_QWORD *)(v23 + 8) & v15) != 0 && v12 < *(_QWORD *)(v23 + 16) )
        v12 = *(_QWORD *)(v23 + 16);
      v33 = v24 + 1;
      if ( v24 + 1 == v22 )
        break;
      while ( 1 )
      {
        v23 = *v33;
        v24 = v33;
        if ( *v33 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v22 == ++v33 )
          goto LABEL_16;
      }
    }
    while ( v22 != v33 );
  }
LABEL_16:
  v25 = *(_DWORD *)(a1 + 152);
  v26 = a1 + 128;
  if ( !v25 )
  {
    ++*(_QWORD *)(a1 + 128);
    goto LABEL_47;
  }
  v27 = 1;
  v28 = *(_QWORD *)(a1 + 136);
  v29 = 0;
  v30 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v31 = (__int64 *)(v28 + 16LL * v30);
  v32 = *v31;
  if ( a2 == *v31 )
    return v31[1];
  while ( v32 != -8 )
  {
    if ( v32 == -16 && !v29 )
      v29 = v31;
    v30 = (v25 - 1) & (v27 + v30);
    v31 = (__int64 *)(v28 + 16LL * v30);
    v32 = *v31;
    if ( a2 == *v31 )
      return v31[1];
    ++v27;
  }
  if ( !v29 )
    v29 = v31;
  v35 = *(_DWORD *)(a1 + 144);
  ++*(_QWORD *)(a1 + 128);
  v36 = v35 + 1;
  if ( 4 * v36 >= 3 * v25 )
  {
LABEL_47:
    sub_390F0C0(v26, 2 * v25);
    v37 = *(_DWORD *)(a1 + 152);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 136);
      v40 = v38 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v36 = *(_DWORD *)(a1 + 144) + 1;
      v29 = (__int64 *)(v39 + 16LL * v40);
      v41 = *v29;
      if ( a2 != *v29 )
      {
        v42 = 1;
        v43 = 0;
        while ( v41 != -8 )
        {
          if ( !v43 && v41 == -16 )
            v43 = v29;
          v40 = v38 & (v42 + v40);
          v29 = (__int64 *)(v39 + 16LL * v40);
          v41 = *v29;
          if ( a2 == *v29 )
            goto LABEL_43;
          ++v42;
        }
        if ( v43 )
          v29 = v43;
      }
      goto LABEL_43;
    }
    goto LABEL_70;
  }
  if ( v25 - *(_DWORD *)(a1 + 148) - v36 <= v25 >> 3 )
  {
    sub_390F0C0(v26, v25);
    v44 = *(_DWORD *)(a1 + 152);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = 1;
      v47 = 0;
      LODWORD(v48) = v45 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v49 = *(_QWORD *)(a1 + 136);
      v36 = *(_DWORD *)(a1 + 144) + 1;
      v29 = (__int64 *)(v49 + 16LL * (unsigned int)v48);
      v50 = *v29;
      if ( a2 != *v29 )
      {
        while ( v50 != -8 )
        {
          if ( !v47 && v50 == -16 )
            v47 = v29;
          v48 = v45 & (unsigned int)(v48 + v46);
          v29 = (__int64 *)(v49 + 16 * v48);
          v50 = *v29;
          if ( a2 == *v29 )
            goto LABEL_43;
          ++v46;
        }
        if ( v47 )
          v29 = v47;
      }
      goto LABEL_43;
    }
LABEL_70:
    ++*(_DWORD *)(a1 + 144);
    BUG();
  }
LABEL_43:
  *(_DWORD *)(a1 + 144) = v36;
  if ( *v29 != -8 )
    --*(_DWORD *)(a1 + 148);
  *v29 = a2;
  v29[1] = v12;
  return v12;
}
