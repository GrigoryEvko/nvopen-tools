// Function: sub_19C26F0
// Address: 0x19c26f0
//
bool __fastcall sub_19C26F0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *i; // r12
  __int64 v14; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // r9d
  __int64 *v20; // rdx
  int v21; // eax
  int v22; // ecx
  int v23; // eax
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // eax
  __int64 v27; // rdi
  int v28; // r10d
  __int64 *v29; // r9
  int v30; // eax
  int v31; // eax
  __int64 v32; // r8
  int v33; // r9d
  __int64 *v34; // rdi
  unsigned int v35; // r14d
  __int64 v36; // rsi

  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_31;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v6 + 112LL * v8);
  v10 = *v9;
  if ( *v9 != a2 )
  {
    v19 = 1;
    v20 = 0;
    while ( v10 != -8 )
    {
      if ( !v20 && v10 == -16 )
        v20 = v9;
      v8 = (v5 - 1) & (v19 + v8);
      v9 = (__int64 *)(v6 + 112LL * v8);
      v10 = *v9;
      if ( *v9 == a2 )
        goto LABEL_3;
      ++v19;
    }
    v21 = *(_DWORD *)(a1 + 16);
    if ( !v20 )
      v20 = v9;
    ++*(_QWORD *)a1;
    v22 = v21 + 1;
    if ( 4 * (v21 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 20) - v22 > v5 >> 3 )
      {
LABEL_27:
        *(_DWORD *)(a1 + 16) = v22;
        if ( *v20 != -8 )
          --*(_DWORD *)(a1 + 20);
        v12 = v20 + 6;
        *v20 = a2;
        v20[1] = 0;
        i = v20 + 6;
        v20[2] = (__int64)(v20 + 6);
        v20[3] = (__int64)(v20 + 6);
        v20[4] = 8;
        *((_DWORD *)v20 + 10) = 0;
        v16 = v20 + 6;
        goto LABEL_13;
      }
      sub_19C24C0(a1, v5);
      v30 = *(_DWORD *)(a1 + 24);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a1 + 8);
        v33 = 1;
        v34 = 0;
        v35 = v31 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v20 = (__int64 *)(v32 + 112LL * v35);
        v22 = *(_DWORD *)(a1 + 16) + 1;
        v36 = *v20;
        if ( a2 != *v20 )
        {
          while ( v36 != -8 )
          {
            if ( !v34 && v36 == -16 )
              v34 = v20;
            v35 = v31 & (v33 + v35);
            v20 = (__int64 *)(v32 + 112LL * v35);
            v36 = *v20;
            if ( *v20 == a2 )
              goto LABEL_27;
            ++v33;
          }
          if ( v34 )
            v20 = v34;
        }
        goto LABEL_27;
      }
LABEL_60:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_31:
    sub_19C24C0(a1, 2 * v5);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = (__int64 *)(v25 + 112LL * v26);
      v22 = *(_DWORD *)(a1 + 16) + 1;
      v27 = *v20;
      if ( *v20 != a2 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -8 )
        {
          if ( !v29 && v27 == -16 )
            v29 = v20;
          v26 = v24 & (v28 + v26);
          v20 = (__int64 *)(v25 + 112LL * v26);
          v27 = *v20;
          if ( *v20 == a2 )
            goto LABEL_27;
          ++v28;
        }
        if ( v29 )
          v20 = v29;
      }
      goto LABEL_27;
    }
    goto LABEL_60;
  }
LABEL_3:
  v11 = (_QWORD *)v9[3];
  v12 = (_QWORD *)v9[2];
  if ( v11 == v12 )
  {
    for ( i = &v12[*((unsigned int *)v9 + 9)]; i != v12; ++v12 )
    {
      if ( a3 == *v12 )
        break;
    }
    v16 = i;
  }
  else
  {
    i = &v11[*((unsigned int *)v9 + 8)];
    v12 = sub_16CC9F0((__int64)(v9 + 1), a3);
    if ( a3 == *v12 )
    {
      v17 = v9[3];
      if ( v17 == v9[2] )
        v18 = *((unsigned int *)v9 + 9);
      else
        v18 = *((unsigned int *)v9 + 8);
      v16 = (_QWORD *)(v17 + 8 * v18);
    }
    else
    {
      v14 = v9[3];
      if ( v14 != v9[2] )
      {
        v12 = (_QWORD *)(v14 + 8LL * *((unsigned int *)v9 + 8));
        return v12 != i;
      }
      v12 = (_QWORD *)(v14 + 8LL * *((unsigned int *)v9 + 9));
      v16 = v12;
    }
  }
LABEL_13:
  while ( v16 != v12 )
  {
    if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v12;
  }
  return v12 != i;
}
