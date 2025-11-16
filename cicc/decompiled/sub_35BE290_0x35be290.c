// Function: sub_35BE290
// Address: 0x35be290
//
__int64 *__fastcall sub_35BE290(__int64 *a1, __int64 a2, unsigned int *a3)
{
  int v6; // r14d
  __int64 v7; // r15
  int v8; // r14d
  unsigned int v9; // ecx
  __int64 *v10; // rsi
  __int64 v11; // rax
  int i; // r8d
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r14
  int v16; // edx
  __int64 v17; // rax
  unsigned __int64 v18; // rbx
  volatile signed __int32 *v19; // rdi
  signed __int32 v20; // eax
  __int64 v21; // rdi
  _DWORD *v22; // rdx
  _DWORD *v23; // rax
  _DWORD *v24; // rdi
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
  v36[0] = (__int64 *)sub_35BA2F0(*((unsigned int **)a3 + 1), (unsigned int *)(*((_QWORD *)a3 + 1) + 4LL * *a3));
  v9 = v8 & sub_C4ECF0((int *)a3, (__int64 *)v36);
  v10 = (__int64 *)(v7 + 8LL * v9);
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
    v22 = *(_DWORD **)(v11 + 32);
    v23 = (_DWORD *)*((_QWORD *)a3 + 1);
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
    v9 = v8 & (i + v9);
    v10 = (__int64 *)(v7 + 8LL * v9);
    v11 = *v10;
    if ( !*v10 )
      goto LABEL_8;
  }
LABEL_24:
  if ( (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) != v10 )
  {
    v21 = *(_QWORD *)(*v10 + 8);
    v25 = *v10 + 24;
    if ( !v21 )
LABEL_49:
      abort();
    v26 = *(_DWORD *)(v21 + 8);
    v27 = (volatile signed __int32 *)(v21 + 8);
    do
    {
      if ( !v26 )
        goto LABEL_49;
      v28 = v26;
      v26 = _InterlockedCompareExchange(v27, v26 + 1, v26);
    }
    while ( v28 != v26 );
    *a1 = v25;
    a1[1] = v21;
    if ( &_pthread_key_create )
      _InterlockedAdd(v27, 1u);
    else
      ++*(_DWORD *)(v21 + 8);
LABEL_31:
    sub_A191D0((volatile signed __int32 *)v21);
    return a1;
  }
LABEL_8:
  v14 = sub_22077B0(0x38u);
  v15 = v14;
  if ( v14 )
  {
    v16 = *a3;
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)(v14 + 8) = 0x100000001LL;
    *(_QWORD *)(v14 + 32) = a2;
    *(_DWORD *)(v14 + 40) = v16;
    *(_QWORD *)v14 = &unk_4A3A3A8;
    v17 = *((_QWORD *)a3 + 1);
    *((_QWORD *)a3 + 1) = 0;
    v18 = v15 + 16;
    *(_QWORD *)(v15 + 48) = v17;
    *(_QWORD *)(v15 + 16) = v15 + 16;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(v15 + 12), 1u);
    else
      *(_DWORD *)(v15 + 12) = 2;
    goto LABEL_11;
  }
  v18 = 16;
  if ( MEMORY[0x18] && *(_DWORD *)(MEMORY[0x18] + 8LL) )
  {
    v34 = 16;
    if ( (unsigned __int8)sub_35BC520(a2, &v34, &v35) )
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
  if ( !(unsigned __int8)sub_35BC520(a2, &v34, &v35) )
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
    sub_35BE110(a2, v30);
    sub_35BC520(a2, &v34, v36);
    v32 = v36[0];
    v33 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_35;
  }
LABEL_17:
  a1[1] = v15;
  *a1 = v18 + 24;
  if ( v15 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(v15 + 8), 1u);
    else
      ++*(_DWORD *)(v15 + 8);
    v21 = v15;
    goto LABEL_31;
  }
  return a1;
}
