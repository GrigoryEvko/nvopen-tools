// Function: sub_1A508F0
// Address: 0x1a508f0
//
void __fastcall sub_1A508F0(_QWORD *src, _QWORD *a2, __int64 a3)
{
  _QWORD *v4; // rsi
  int v7; // edx
  __int64 v8; // r12
  _QWORD *v9; // rdi
  _QWORD *v10; // r14
  int v11; // edx
  __int64 v12; // r9
  unsigned int v13; // r10d
  int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // rcx
  _QWORD **v17; // rax
  _QWORD *v18; // rdi
  unsigned int i; // eax
  unsigned int v20; // r8d
  __int64 *v21; // rdi
  __int64 v22; // r14
  _QWORD **v23; // rdi
  _QWORD *v24; // rdi
  unsigned int j; // r8d
  int v26; // ecx
  int v27; // edi
  __int64 v28; // r8
  unsigned int v29; // esi
  __int64 *v30; // rax
  _QWORD **v31; // rax
  _QWORD *v32; // rcx
  unsigned int k; // eax
  unsigned int v34; // esi
  __int64 *v35; // rcx
  __int64 v36; // r11
  _QWORD **v37; // rdx
  _QWORD *v38; // rdx
  unsigned int m; // ecx
  int v40; // edx
  int v41; // eax
  int v42; // r11d
  __int64 v43; // r11
  int v44; // eax
  int v45; // r8d
  __int64 v46; // rax
  int v47; // r15d
  int v48; // [rsp+4h] [rbp-3Ch]

  if ( src == a2 )
    return;
  v4 = src + 1;
  if ( a2 == src + 1 )
    return;
  do
  {
    v7 = *(_DWORD *)(a3 + 24);
    v8 = *v4;
    v9 = v4;
    v10 = v4 + 1;
    if ( !v7 )
      goto LABEL_21;
    v11 = v7 - 1;
    v12 = *(_QWORD *)(a3 + 8);
    v13 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
    v14 = v11 & v13;
    v15 = (__int64 *)(v12 + 16LL * (v11 & v13));
    v16 = *v15;
    if ( v8 != *v15 )
    {
      v43 = *v15;
      v44 = 1;
      while ( v43 != -8 )
      {
        v45 = v44 + 1;
        v46 = v11 & (unsigned int)(v14 + v44);
        v14 = v46;
        v15 = (__int64 *)(v12 + 16 * v46);
        v43 = *v15;
        if ( v8 == *v15 )
          goto LABEL_5;
        v44 = v45;
      }
LABEL_46:
      i = 0;
      goto LABEL_8;
    }
LABEL_5:
    v17 = (_QWORD **)v15[1];
    if ( !v17 )
      goto LABEL_46;
    v18 = *v17;
    for ( i = 1; v18; ++i )
      v18 = (_QWORD *)*v18;
LABEL_8:
    v20 = v11 & (((unsigned int)*src >> 9) ^ ((unsigned int)*src >> 4));
    v21 = (__int64 *)(v12 + 16LL * v20);
    v22 = *v21;
    if ( *src == *v21 )
    {
LABEL_9:
      v23 = (_QWORD **)v21[1];
      if ( v23 )
      {
        v24 = *v23;
        for ( j = 1; v24; ++j )
          v24 = (_QWORD *)*v24;
        v10 = v4 + 1;
        if ( j > i )
        {
          if ( src != v4 )
            memmove(src + 1, src, (char *)v4 - (char *)src);
          *src = v8;
          goto LABEL_16;
        }
        goto LABEL_25;
      }
    }
    else
    {
      v27 = 1;
      while ( v22 != -8 )
      {
        v47 = v27 + 1;
        v20 = v11 & (v27 + v20);
        v21 = (__int64 *)(v12 + 16LL * v20);
        v22 = *v21;
        if ( *src == *v21 )
          goto LABEL_9;
        v27 = v47;
      }
    }
    v10 = v4 + 1;
LABEL_25:
    v28 = *(v4 - 1);
    v9 = v4 - 1;
    while ( 1 )
    {
      v29 = v11 & v13;
      v30 = (__int64 *)(v12 + 16LL * (v11 & v13));
      if ( v8 != v16 )
      {
        v41 = 1;
        while ( v16 != -8 )
        {
          v42 = v41 + 1;
          v29 = v11 & (v41 + v29);
          v30 = (__int64 *)(v12 + 16LL * v29);
          v16 = *v30;
          if ( v8 == *v30 )
            goto LABEL_27;
          v41 = v42;
        }
LABEL_39:
        k = 0;
        goto LABEL_30;
      }
LABEL_27:
      v31 = (_QWORD **)v30[1];
      if ( !v31 )
        goto LABEL_39;
      v32 = *v31;
      for ( k = 1; v32; ++k )
        v32 = (_QWORD *)*v32;
LABEL_30:
      v34 = v11 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v35 = (__int64 *)(v12 + 16LL * v34);
      v36 = *v35;
      if ( *v35 != v28 )
        break;
LABEL_31:
      v37 = (_QWORD **)v35[1];
      if ( !v37 )
        goto LABEL_20;
      v38 = *v37;
      for ( m = 1; v38; ++m )
        v38 = (_QWORD *)*v38;
      if ( m <= k )
        goto LABEL_20;
      v9[1] = v28;
      v40 = *(_DWORD *)(a3 + 24);
      if ( !v40 )
        goto LABEL_21;
      v11 = v40 - 1;
      v12 = *(_QWORD *)(a3 + 8);
      v28 = *--v9;
      v16 = *(_QWORD *)(v12 + 16LL * (v13 & v11));
    }
    v26 = 1;
    while ( v36 != -8 )
    {
      v34 = v11 & (v26 + v34);
      v48 = v26 + 1;
      v35 = (__int64 *)(v12 + 16LL * v34);
      v36 = *v35;
      if ( *v35 == v28 )
        goto LABEL_31;
      v26 = v48;
    }
LABEL_20:
    ++v9;
LABEL_21:
    *v9 = v8;
LABEL_16:
    v4 = v10;
  }
  while ( a2 != v10 );
}
