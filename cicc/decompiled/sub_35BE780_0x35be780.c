// Function: sub_35BE780
// Address: 0x35be780
//
__int64 *__fastcall sub_35BE780(__int64 *a1, __int64 a2, unsigned int *a3)
{
  int v6; // r14d
  __int64 v7; // r15
  int v8; // r14d
  __int64 v9; // rsi
  __int64 *v10; // rcx
  __int64 v11; // rax
  int i; // r8d
  __int64 v13; // rdi
  _QWORD *v14; // rax
  _QWORD *v15; // r14
  int v16; // edx
  __int64 v17; // rax
  unsigned __int64 v18; // rbx
  volatile signed __int32 *v19; // rdi
  signed __int32 v20; // eax
  _QWORD *v21; // rdi
  float *v22; // rdx
  float *v23; // rax
  float *v24; // rdi
  __int64 v25; // rcx
  signed __int32 v26; // eax
  volatile signed __int32 *v27; // rdx
  signed __int32 v28; // ett
  unsigned int v30; // esi
  int v31; // eax
  __int64 *v32; // rdx
  int v33; // eax
  unsigned __int64 v34; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v35; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v36[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 8);
  if ( !v6 )
    goto LABEL_8;
  v8 = v6 - 1;
  v36[0] = (__int64 *)sub_25FD4A0(*((_QWORD **)a3 + 1), *((_QWORD *)a3 + 1) + 4LL * *a3);
  v9 = v8 & (unsigned int)sub_C4ECF0((int *)a3, (__int64 *)v36);
  v10 = (__int64 *)(v7 + 8 * v9);
  v11 = *v10;
  if ( !*v10 )
    goto LABEL_8;
  for ( i = 1; ; ++i )
  {
    if ( v11 == 1 )
      goto LABEL_6;
    v13 = *a3;
    if ( (_DWORD)v13 != *(_DWORD *)(v11 + 24) )
      goto LABEL_6;
    v22 = *(float **)(v11 + 32);
    v23 = (float *)*((_QWORD *)a3 + 1);
    v24 = &v23[v13];
    if ( v23 == v24 )
      break;
    while ( *v23 == *v22 )
    {
      ++v23;
      ++v22;
      if ( v24 == v23 )
        goto LABEL_24;
    }
LABEL_6:
    LODWORD(v9) = v8 & (i + v9);
    v10 = (__int64 *)(v7 + 8LL * (unsigned int)v9);
    v11 = *v10;
    if ( !*v10 )
      goto LABEL_8;
  }
LABEL_24:
  if ( (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) != v10 )
  {
    v21 = *(_QWORD **)(*v10 + 8);
    v25 = *v10 + 24;
    if ( !v21 )
LABEL_49:
      abort();
    v26 = *((_DWORD *)v21 + 2);
    v27 = (volatile signed __int32 *)(v21 + 1);
    do
    {
      if ( !v26 )
        goto LABEL_49;
      v28 = v26;
      v26 = _InterlockedCompareExchange(v27, v26 + 1, v26);
    }
    while ( v28 != v26 );
    *a1 = v25;
    a1[1] = (__int64)v21;
    if ( &_pthread_key_create )
      _InterlockedAdd(v27, 1u);
    else
      ++*((_DWORD *)v21 + 2);
LABEL_31:
    sub_A191D0((volatile signed __int32 *)v21);
    return a1;
  }
LABEL_8:
  v14 = (_QWORD *)sub_22077B0(0x38u);
  v15 = v14;
  if ( v14 )
  {
    v16 = *a3;
    *a3 = 0;
    v14[1] = 0x100000001LL;
    v14[3] = 0;
    v14[4] = a2;
    *v14 = &unk_4A3A3E0;
    v17 = *((_QWORD *)a3 + 1);
    *((_QWORD *)a3 + 1) = 0;
    v18 = (unsigned __int64)(v15 + 2);
    *((_DWORD *)v15 + 10) = v16;
    v15[6] = v17;
    v15[2] = v15 + 2;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)v15 + 3, 1u);
    else
      *((_DWORD *)v15 + 3) = 2;
    goto LABEL_11;
  }
  v18 = 16;
  if ( MEMORY[0x18] && *(_DWORD *)(MEMORY[0x18] + 8LL) )
  {
    v34 = 16;
    if ( (unsigned __int8)sub_35BD940(a2, &v34, &v35) )
    {
      *a1 = 40;
      a1[1] = 0;
      return a1;
    }
    goto LABEL_33;
  }
  MEMORY[0x10] = 16;
LABEL_11:
  v19 = *(volatile signed __int32 **)(v18 + 8);
  if ( v19 )
  {
    if ( &_pthread_key_create )
    {
      v20 = _InterlockedExchangeAdd(v19 + 3, 0xFFFFFFFF);
    }
    else
    {
      v20 = *((_DWORD *)v19 + 3);
      *((_DWORD *)v19 + 3) = v20 - 1;
    }
    if ( v20 == 1 )
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 24LL))(v19);
  }
  *(_QWORD *)(v18 + 8) = v15;
  v34 = v18;
  if ( !(unsigned __int8)sub_35BD940(a2, &v34, &v35) )
  {
LABEL_33:
    v30 = *(_DWORD *)(a2 + 24);
    v31 = *(_DWORD *)(a2 + 16);
    v32 = v35;
    ++*(_QWORD *)a2;
    v33 = v31 + 1;
    v36[0] = v32;
    if ( 4 * v33 >= 3 * v30 )
    {
      v30 *= 2;
    }
    else if ( v30 - *(_DWORD *)(a2 + 20) - v33 > v30 >> 3 )
    {
LABEL_35:
      *(_DWORD *)(a2 + 16) = v33;
      if ( *v32 )
        --*(_DWORD *)(a2 + 20);
      *v32 = v34;
      goto LABEL_17;
    }
    sub_35BE600(a2, v30);
    sub_35BD940(a2, &v34, v36);
    v32 = v36[0];
    v33 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_35;
  }
LABEL_17:
  a1[1] = (__int64)v15;
  *a1 = v18 + 24;
  if ( v15 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)v15 + 2, 1u);
    else
      ++*((_DWORD *)v15 + 2);
    v21 = v15;
    goto LABEL_31;
  }
  return a1;
}
