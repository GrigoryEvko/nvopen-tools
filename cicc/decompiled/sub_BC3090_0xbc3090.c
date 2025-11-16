// Function: sub_BC3090
// Address: 0xbc3090
//
int __fastcall sub_BC3090(pthread_rwlock_t *rwlock, _QWORD *a2, char a3)
{
  unsigned int v5; // esi
  __int64 v6; // r13
  __int64 v7; // r9
  unsigned int v8; // edi
  _QWORD *v9; // rax
  __int64 v10; // rcx
  const void *v11; // r13
  size_t v12; // rbx
  unsigned int v13; // eax
  __int64 v14; // rdx
  void (*pad2)(); // rax
  unsigned int v16; // r10d
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 pad1; // r13
  unsigned __int64 i; // r15
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned int v23; // r10d
  _QWORD *v24; // rcx
  _QWORD *v25; // r11
  char *v26; // rsi
  int v27; // r8d
  _QWORD *v28; // rdx
  int writer; // eax
  int v30; // ecx
  int v31; // edi
  int v32; // edi
  __int64 v33; // r9
  unsigned int v34; // eax
  __int64 v35; // rsi
  int v36; // r11d
  _QWORD *v37; // r10
  int v38; // eax
  int v39; // esi
  __int64 v40; // rdi
  _QWORD *v41; // r9
  unsigned int v42; // r15d
  int v43; // r10d
  __int64 v44; // rax
  _QWORD *v46; // [rsp+8h] [rbp-58h]
  _QWORD *v47; // [rsp+10h] [rbp-50h]
  unsigned int v48; // [rsp+18h] [rbp-48h]
  __int64 v50[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( &_pthread_key_create && pthread_rwlock_wrlock(rwlock) == 35 )
    sub_4264C5(0x23u);
  v5 = *((_DWORD *)&rwlock[1].__align + 8);
  v6 = a2[4];
  if ( !v5 )
  {
    ++*(&rwlock[1].__align + 1);
    goto LABEL_42;
  }
  v7 = *(&rwlock[1].__align + 2);
  v8 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v9 = (_QWORD *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( v6 == *v9 )
    goto LABEL_5;
  v27 = 1;
  v28 = 0;
  while ( v10 != -4096 )
  {
    if ( v10 != -8192 || v28 )
      v9 = v28;
    v8 = (v5 - 1) & (v27 + v8);
    v10 = *(_QWORD *)(v7 + 16LL * v8);
    if ( v6 == v10 )
      goto LABEL_5;
    ++v27;
    v28 = v9;
    v9 = (_QWORD *)(v7 + 16LL * v8);
  }
  if ( !v28 )
    v28 = v9;
  writer = rwlock[1].__writer;
  ++*(&rwlock[1].__align + 1);
  v30 = writer + 1;
  if ( 4 * (writer + 1) >= 3 * v5 )
  {
LABEL_42:
    sub_B858F0((__int64)(&rwlock[1].__align + 1), 2 * v5);
    v31 = *((_DWORD *)&rwlock[1].__align + 8);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(&rwlock[1].__align + 2);
      v34 = v32 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v30 = rwlock[1].__writer + 1;
      v28 = (_QWORD *)(v33 + 16LL * v34);
      v35 = *v28;
      if ( v6 != *v28 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v37 )
            v37 = v28;
          v34 = v32 & (v36 + v34);
          v28 = (_QWORD *)(v33 + 16LL * v34);
          v35 = *v28;
          if ( v6 == *v28 )
            goto LABEL_36;
          ++v36;
        }
        if ( v37 )
          v28 = v37;
      }
      goto LABEL_36;
    }
    goto LABEL_71;
  }
  if ( v5 - rwlock[1].__shared - v30 <= v5 >> 3 )
  {
    sub_B858F0((__int64)(&rwlock[1].__align + 1), v5);
    v38 = *((_DWORD *)&rwlock[1].__align + 8);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(&rwlock[1].__align + 2);
      v41 = 0;
      v42 = (v38 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v43 = 1;
      v30 = rwlock[1].__writer + 1;
      v28 = (_QWORD *)(v40 + 16LL * v42);
      v44 = *v28;
      if ( v6 != *v28 )
      {
        while ( v44 != -4096 )
        {
          if ( !v41 && v44 == -8192 )
            v41 = v28;
          v42 = v39 & (v43 + v42);
          v28 = (_QWORD *)(v40 + 16LL * v42);
          v44 = *v28;
          if ( v6 == *v28 )
            goto LABEL_36;
          ++v43;
        }
        if ( v41 )
          v28 = v41;
      }
      goto LABEL_36;
    }
LABEL_71:
    ++rwlock[1].__writer;
    BUG();
  }
LABEL_36:
  rwlock[1].__writer = v30;
  if ( *v28 != -4096 )
    --rwlock[1].__shared;
  *v28 = v6;
  v28[1] = a2;
LABEL_5:
  v11 = (const void *)a2[2];
  v12 = a2[3];
  v13 = sub_C92610(v11, v12);
  v14 = (unsigned int)sub_C92740(&rwlock[1].__align + 5, v11, v12, v13);
  pad2 = (void (*)())rwlock[1].__pad2;
  v16 = v14;
  v17 = (_QWORD *)((char *)pad2 + 8 * v14);
  v18 = *v17;
  if ( *v17 )
  {
    if ( v18 != -8 )
      goto LABEL_7;
    --rwlock[2].__lock;
  }
  v47 = v17;
  v48 = v16;
  v22 = sub_C7D670(v12 + 17, 8);
  v23 = v48;
  v24 = v47;
  v25 = (_QWORD *)v22;
  if ( v12 )
  {
    v46 = (_QWORD *)v22;
    memcpy((void *)(v22 + 16), v11, v12);
    v23 = v48;
    v24 = v47;
    v25 = v46;
  }
  *((_BYTE *)v25 + v12 + 16) = 0;
  *v25 = v12;
  v25[1] = 0;
  *v24 = v25;
  ++*((_DWORD *)&rwlock[1].__align + 13);
  pad2 = (void (*)())(rwlock[1].__pad2 + 8LL * (unsigned int)sub_C929D0(&rwlock[1].__align + 5, v23));
  v18 = *(_QWORD *)pad2;
  if ( *(_QWORD *)pad2 == -8 || !v18 )
  {
    pad2 = (void (*)())((char *)pad2 + 8);
    do
    {
      do
      {
        v18 = *(_QWORD *)pad2;
        pad2 = (void (*)())((char *)pad2 + 8);
      }
      while ( !v18 );
    }
    while ( v18 == -8 );
  }
LABEL_7:
  *(_QWORD *)(v18 + 8) = a2;
  pad1 = rwlock[2].__pad1;
  for ( i = rwlock[2].__pad2; i != pad1; LODWORD(pad2) = ((__int64 (__fastcall *)(__int64, _QWORD *))pad2)(v21, a2) )
  {
    while ( 1 )
    {
      v21 = *(_QWORD *)pad1;
      pad2 = *(void (**)())(**(_QWORD **)pad1 + 16LL);
      if ( pad2 != nullsub_78 )
        break;
      pad1 += 8LL;
      if ( i == pad1 )
        goto LABEL_12;
    }
    pad1 += 8LL;
  }
LABEL_12:
  if ( !a3 )
  {
LABEL_13:
    if ( !&_pthread_key_create )
      return (int)pad2;
    goto LABEL_14;
  }
  v50[0] = (__int64)a2;
  v26 = (char *)*(&rwlock[2].__align + 2);
  if ( v26 == *((char **)&rwlock[2].__align + 3) )
  {
    LODWORD(pad2) = sub_BC2EE0(&rwlock[2].__align + 1, v26, v50);
    a2 = (_QWORD *)v50[0];
    if ( !v50[0] )
      goto LABEL_13;
LABEL_29:
    LODWORD(pad2) = j_j___libc_free_0(a2, 56);
    goto LABEL_13;
  }
  if ( !v26 )
  {
    *(&rwlock[2].__align + 2) = 8;
    goto LABEL_29;
  }
  *(_QWORD *)v26 = a2;
  *(&rwlock[2].__align + 2) += 8;
  if ( !&_pthread_key_create )
    return (int)pad2;
LABEL_14:
  LODWORD(pad2) = pthread_rwlock_unlock(rwlock);
  return (int)pad2;
}
