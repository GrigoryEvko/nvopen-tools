// Function: sub_239DED0
// Address: 0x239ded0
//
__int64 __fastcall sub_239DED0(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // r13
  unsigned int v8; // edx
  char v9; // al
  __int64 v10; // rdx
  __int64 *v11; // r15
  __int64 *v12; // r12
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rdx
  _BYTE *v16; // r15
  _BYTE *v17; // r13
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // eax
  __int64 *v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rax
  unsigned __int64 *v24; // rax
  unsigned __int64 v25; // r12
  unsigned int v26; // eax
  __int64 v27; // rcx
  int v28; // edx
  int v30; // r8d
  __int64 v31; // rax
  __int64 v33; // [rsp+18h] [rbp-88h]
  _BYTE *v34; // [rsp+40h] [rbp-60h] BYREF
  __int64 v35; // [rsp+48h] [rbp-58h]
  _BYTE v36[80]; // [rsp+50h] [rbp-50h] BYREF

  v7 = (__int64 *)a4;
  v34 = v36;
  v8 = *(_DWORD *)(a1 + 24);
  v35 = 0x400000000LL;
  v33 = (__int64)a2;
  v9 = *(_BYTE *)(a1 + 24) & 1;
  v10 = v8 >> 1;
  if ( (_DWORD)v10 )
  {
    if ( v9 )
    {
      v11 = (__int64 *)(a1 + 32);
      v12 = (__int64 *)(a1 + 64);
      do
      {
LABEL_4:
        v10 = *v11;
        if ( *v11 != -4096 && v10 != -8192 )
          break;
        v11 += 2;
      }
      while ( v11 != v12 );
      goto LABEL_7;
    }
    v11 = *(__int64 **)(a1 + 32);
    a5 = 2LL * *(unsigned int *)(a1 + 40);
    v12 = &v11[a5];
    if ( v11 != &v11[a5] )
      goto LABEL_4;
  }
  else
  {
    if ( v9 )
    {
      v10 = a1 + 32;
      v31 = 32;
    }
    else
    {
      v10 = *(_QWORD *)(a1 + 32);
      v31 = 16LL * *(unsigned int *)(a1 + 40);
    }
    v11 = (__int64 *)(v10 + v31);
    v12 = (__int64 *)(v10 + v31);
  }
LABEL_7:
  if ( v12 == v11 )
    return 0;
  do
  {
    v13 = *v11;
    sub_239DC20(v11 + 1, (__int64)a2, v10, a4, a5 * 8, a6, v7, v33, a3);
    v14 = v11[1];
    a4 = v14 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v14 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v14 & 4) != 0 && !*(_DWORD *)(a4 + 8) )
    {
      v15 = (unsigned int)v35;
      if ( (unsigned __int64)(unsigned int)v35 + 1 > HIDWORD(v35) )
      {
        a2 = v36;
        sub_C8D5F0((__int64)&v34, v36, (unsigned int)v35 + 1LL, 8u, a5 * 8, a6);
        v15 = (unsigned int)v35;
      }
      a4 = (unsigned __int64)v34;
      *(_QWORD *)&v34[8 * v15] = v13;
      LODWORD(v35) = v35 + 1;
    }
    v11 += 2;
    if ( v11 == v12 )
      break;
    while ( 1 )
    {
      v10 = *v11;
      if ( *v11 != -8192 && v10 != -4096 )
        break;
      v11 += 2;
      if ( v12 == v11 )
        goto LABEL_18;
    }
  }
  while ( v12 != v11 );
LABEL_18:
  v16 = v34;
  v17 = &v34[8 * (unsigned int)v35];
  if ( v17 == v34 )
    goto LABEL_37;
  do
  {
    v27 = *(_QWORD *)v16;
    if ( (*(_BYTE *)(a1 + 24) & 1) != 0 )
    {
      v18 = a1 + 32;
      v19 = 1;
    }
    else
    {
      v28 = *(_DWORD *)(a1 + 40);
      v18 = *(_QWORD *)(a1 + 32);
      if ( !v28 )
        goto LABEL_29;
      v19 = v28 - 1;
    }
    v20 = v19 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
    v21 = (__int64 *)(v18 + 16LL * v20);
    v22 = *v21;
    if ( v27 == *v21 )
    {
LABEL_22:
      v23 = v21[1];
      if ( v23 )
      {
        if ( (v23 & 4) != 0 )
        {
          v24 = (unsigned __int64 *)(v23 & 0xFFFFFFFFFFFFFFF8LL);
          v25 = (unsigned __int64)v24;
          if ( v24 )
          {
            if ( (unsigned __int64 *)*v24 != v24 + 2 )
              _libc_free(*v24);
            j_j___libc_free_0(v25);
          }
        }
      }
      *v21 = -8192;
      v26 = *(_DWORD *)(a1 + 24);
      ++*(_DWORD *)(a1 + 28);
      *(_DWORD *)(a1 + 24) = (2 * (v26 >> 1) - 2) | v26 & 1;
    }
    else
    {
      v30 = 1;
      while ( v22 != -4096 )
      {
        v20 = v19 & (v30 + v20);
        v21 = (__int64 *)(v18 + 16LL * v20);
        v22 = *v21;
        if ( v27 == *v21 )
          goto LABEL_22;
        ++v30;
      }
    }
LABEL_29:
    v16 += 8;
  }
  while ( v17 != v16 );
  v16 = v34;
LABEL_37:
  if ( v16 != v36 )
    _libc_free((unsigned __int64)v16);
  return 0;
}
