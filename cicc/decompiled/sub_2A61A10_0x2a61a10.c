// Function: sub_2A61A10
// Address: 0x2a61a10
//
bool __fastcall sub_2A61A10(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, __int64 a5)
{
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // r11d
  unsigned int v12; // r8d
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned int *v17; // rcx
  _QWORD *v18; // r13
  unsigned int *v19; // r15
  int v20; // eax
  bool result; // al
  int v22; // edi
  int v23; // edi
  unsigned int *v24; // rax
  __int64 v25; // rax
  unsigned int *v26; // rdx
  char v27; // di
  int v28; // eax
  int v29; // r8d
  __int64 v30; // rsi
  unsigned int v31; // edx
  __int64 v32; // rcx
  int v33; // r10d
  _QWORD *v34; // r9
  int v35; // eax
  int v36; // edx
  __int64 v37; // r8
  int v38; // r10d
  unsigned int v39; // ecx
  __int64 v40; // rsi
  unsigned int *v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+10h] [rbp-40h]
  unsigned int *v43; // [rsp+10h] [rbp-40h]

  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_42;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (_QWORD *)(v10 + 56LL * v12);
  v14 = 0;
  v15 = *v13;
  if ( *v13 != a2 )
  {
    while ( v15 != -4096 )
    {
      if ( v15 == -8192 && !v14 )
        v14 = v13;
      v12 = (v9 - 1) & (v11 + v12);
      v13 = (_QWORD *)(v10 + 56LL * v12);
      v15 = *v13;
      if ( *v13 == a2 )
        goto LABEL_3;
      ++v11;
    }
    v22 = *(_DWORD *)(a1 + 16);
    if ( !v14 )
      v14 = v13;
    ++*(_QWORD *)a1;
    v23 = v22 + 1;
    if ( 4 * v23 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 20) - v23 > v9 >> 3 )
      {
LABEL_28:
        *(_DWORD *)(a1 + 16) = v23;
        if ( *v14 != -4096 )
          --*(_DWORD *)(a1 + 20);
        v19 = (unsigned int *)(v14 + 2);
        *((_DWORD *)v14 + 4) = 0;
        v14[3] = 0;
        v17 = (unsigned int *)(v14 + 2);
        v14[4] = v14 + 2;
        v14[5] = v14 + 2;
        v14[6] = 0;
        *v14 = a2;
        v18 = v14 + 1;
        goto LABEL_31;
      }
      sub_2A61720(a1, v9);
      v35 = *(_DWORD *)(a1 + 24);
      if ( v35 )
      {
        v36 = v35 - 1;
        v37 = *(_QWORD *)(a1 + 8);
        v34 = 0;
        v38 = 1;
        v39 = (v35 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v23 = *(_DWORD *)(a1 + 16) + 1;
        v14 = (_QWORD *)(v37 + 56LL * v39);
        v40 = *v14;
        if ( *v14 == a2 )
          goto LABEL_28;
        while ( v40 != -4096 )
        {
          if ( v40 == -8192 && !v34 )
            v34 = v14;
          v39 = v36 & (v38 + v39);
          v14 = (_QWORD *)(v37 + 56LL * v39);
          v40 = *v14;
          if ( *v14 == a2 )
            goto LABEL_28;
          ++v38;
        }
        goto LABEL_46;
      }
      goto LABEL_62;
    }
LABEL_42:
    sub_2A61720(a1, 2 * v9);
    v28 = *(_DWORD *)(a1 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 8);
      v31 = (v28 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 16) + 1;
      v14 = (_QWORD *)(v30 + 56LL * v31);
      v32 = *v14;
      if ( *v14 == a2 )
        goto LABEL_28;
      v33 = 1;
      v34 = 0;
      while ( v32 != -4096 )
      {
        if ( !v34 && v32 == -8192 )
          v34 = v14;
        v31 = v29 & (v33 + v31);
        v14 = (_QWORD *)(v30 + 56LL * v31);
        v32 = *v14;
        if ( *v14 == a2 )
          goto LABEL_28;
        ++v33;
      }
LABEL_46:
      if ( v34 )
        v14 = v34;
      goto LABEL_28;
    }
LABEL_62:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_3:
  v16 = v13[3];
  v17 = (unsigned int *)(v13 + 2);
  v18 = v13 + 1;
  v19 = (unsigned int *)(v13 + 2);
  if ( !v16 )
  {
LABEL_31:
    v42 = (__int64)v19;
    v41 = v17;
    v24 = (unsigned int *)sub_22077B0(0x30u);
    v24[8] = a3;
    v19 = v24;
    v24[9] = a4;
    v24[10] = 0;
    v25 = sub_2A615C0(v18, v42, v24 + 8);
    if ( v26 )
    {
      v27 = 1;
      if ( v41 != v26 && !v25 && a3 >= v26[8] )
      {
        v27 = 0;
        if ( a3 == v26[8] )
          v27 = a4 < v26[9];
      }
      sub_220F040(v27, (__int64)v19, v26, v41);
      ++v18[5];
    }
    else
    {
      v43 = (unsigned int *)v25;
      j_j___libc_free_0((unsigned __int64)v19);
      v19 = v43;
    }
    goto LABEL_13;
  }
  do
  {
    while ( 1 )
    {
      if ( a3 > *(_DWORD *)(v16 + 32) )
      {
        v16 = *(_QWORD *)(v16 + 24);
        goto LABEL_9;
      }
      if ( a3 == *(_DWORD *)(v16 + 32) && a4 > *(_DWORD *)(v16 + 36) )
        break;
      v19 = (unsigned int *)v16;
      v16 = *(_QWORD *)(v16 + 16);
      if ( !v16 )
        goto LABEL_10;
    }
    v16 = *(_QWORD *)(v16 + 24);
LABEL_9:
    ;
  }
  while ( v16 );
LABEL_10:
  if ( v19 == v17 || a3 < v19[8] || a3 == v19[8] && a4 < v19[9] )
    goto LABEL_31;
LABEL_13:
  v20 = v19[10] + 1;
  v19[10] = v20;
  result = v20 == 1;
  if ( result )
    *(_QWORD *)(a1 + 32) += a5;
  return result;
}
