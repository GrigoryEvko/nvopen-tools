// Function: sub_1ED0750
// Address: 0x1ed0750
//
__int64 *__fastcall sub_1ED0750(__int64 *a1, __int64 a2, unsigned int *a3)
{
  int v6; // r15d
  _QWORD *v7; // rax
  _QWORD *v8; // r15
  int v9; // edx
  __int64 v10; // rax
  unsigned __int64 v11; // r12
  volatile signed __int32 *v12; // rdi
  signed __int32 v13; // eax
  char v14; // al
  __int64 *v15; // rdx
  _QWORD *v16; // rdi
  int v18; // r15d
  unsigned int v19; // esi
  __int64 *v20; // rcx
  __int64 v21; // rax
  int i; // r9d
  __int64 v23; // rdi
  float *v24; // rdx
  float *v25; // rax
  float *v26; // rdi
  unsigned int v27; // esi
  int v28; // eax
  int v29; // eax
  __int64 v30; // rcx
  signed __int32 v31; // eax
  volatile signed __int32 *v32; // rdx
  signed __int32 v33; // ett
  __int64 v34; // [rsp+8h] [rbp-48h]
  unsigned __int64 v35; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v36[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a2 + 24);
  v34 = *(_QWORD *)(a2 + 8);
  if ( !v6 )
    goto LABEL_2;
  v18 = v6 - 1;
  v36[0] = (__int64 *)sub_1ECD5B0(*((_QWORD **)a3 + 1), *((_QWORD *)a3 + 1) + 4LL * *a3);
  v19 = v18 & sub_18FDAA0((int *)a3, (__int64 *)v36);
  v20 = (__int64 *)(v34 + 8LL * v19);
  v21 = *v20;
  if ( !*v20 )
    goto LABEL_2;
  for ( i = 1; ; ++i )
  {
    if ( v21 == 1 )
      goto LABEL_26;
    v23 = *a3;
    if ( (_DWORD)v23 != *(_DWORD *)(v21 + 24) )
      goto LABEL_26;
    v24 = *(float **)(v21 + 32);
    v25 = (float *)*((_QWORD *)a3 + 1);
    v26 = &v25[v23];
    if ( v25 == v26 )
      break;
    while ( *v25 == *v24 )
    {
      ++v25;
      ++v24;
      if ( v26 == v25 )
        goto LABEL_33;
    }
LABEL_26:
    v19 = v18 & (i + v19);
    v20 = (__int64 *)(v34 + 8LL * v19);
    v21 = *v20;
    if ( !*v20 )
      goto LABEL_2;
  }
LABEL_33:
  if ( (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) != v20 )
  {
    v16 = *(_QWORD **)(*v20 + 8);
    v30 = *v20 + 24;
    if ( !v16 )
LABEL_51:
      abort();
    v31 = *((_DWORD *)v16 + 2);
    v32 = (volatile signed __int32 *)(v16 + 1);
    do
    {
      if ( !v31 )
        goto LABEL_51;
      v33 = v31;
      v31 = _InterlockedCompareExchange(v32, v31 + 1, v31);
    }
    while ( v33 != v31 );
    *a1 = v30;
    a1[1] = (__int64)v16;
    if ( &_pthread_key_create )
      _InterlockedAdd(v32, 1u);
    else
      ++*((_DWORD *)v16 + 2);
LABEL_16:
    sub_A191D0((volatile signed __int32 *)v16);
    return a1;
  }
LABEL_2:
  v7 = (_QWORD *)sub_22077B0(56);
  v8 = v7;
  if ( v7 )
  {
    v9 = *a3;
    *a3 = 0;
    v7[1] = 0x100000001LL;
    v7[3] = 0;
    v7[4] = a2;
    *v7 = &unk_49FDED0;
    v10 = *((_QWORD *)a3 + 1);
    *((_QWORD *)a3 + 1) = 0;
    v11 = (unsigned __int64)(v8 + 2);
    *((_DWORD *)v8 + 10) = v9;
    v8[6] = v10;
    v8[2] = v8 + 2;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)v8 + 3, 1u);
    else
      *((_DWORD *)v8 + 3) = 2;
    goto LABEL_5;
  }
  v11 = 16;
  if ( !MEMORY[0x18] || !*(_DWORD *)(MEMORY[0x18] + 8LL) )
  {
    MEMORY[0x10] = 16;
LABEL_5:
    v12 = *(volatile signed __int32 **)(v11 + 8);
    if ( v12 )
    {
      if ( &_pthread_key_create )
      {
        v13 = _InterlockedExchangeAdd(v12 + 3, 0xFFFFFFFF);
      }
      else
      {
        v13 = *((_DWORD *)v12 + 3);
        *((_DWORD *)v12 + 3) = v13 - 1;
      }
      if ( v13 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v12 + 24LL))(v12);
    }
    *(_QWORD *)(v11 + 8) = v8;
  }
  v35 = v11;
  v14 = sub_1ED01F0(a2, &v35, v36);
  v15 = v36[0];
  if ( !v14 )
  {
    v27 = *(_DWORD *)(a2 + 24);
    v28 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v29 = v28 + 1;
    if ( 4 * v29 >= 3 * v27 )
    {
      v27 *= 2;
    }
    else if ( v27 - *(_DWORD *)(a2 + 20) - v29 > v27 >> 3 )
    {
LABEL_30:
      *(_DWORD *)(a2 + 16) = v29;
      if ( *v15 )
        --*(_DWORD *)(a2 + 20);
      *v15 = v35;
      goto LABEL_12;
    }
    sub_1ED05F0(a2, v27);
    sub_1ED01F0(a2, &v35, v36);
    v15 = v36[0];
    v29 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_30;
  }
LABEL_12:
  a1[1] = (__int64)v8;
  *a1 = v11 + 24;
  if ( v8 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)v8 + 2, 1u);
    else
      ++*((_DWORD *)v8 + 2);
    v16 = v8;
    goto LABEL_16;
  }
  return a1;
}
