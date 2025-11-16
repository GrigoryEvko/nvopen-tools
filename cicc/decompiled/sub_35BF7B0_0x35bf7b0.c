// Function: sub_35BF7B0
// Address: 0x35bf7b0
//
__int64 *__fastcall sub_35BF7B0(__int64 *a1, __int64 a2, __int64 *a3)
{
  int v6; // r14d
  __int64 v7; // r15
  int v8; // r14d
  unsigned int v9; // ecx
  __int64 *v10; // rsi
  __int64 v11; // rax
  int i; // r8d
  _QWORD *v13; // rax
  volatile signed __int32 *v14; // r14
  __int64 v15; // rdx
  __int64 v16; // r15
  unsigned int *v17; // rdi
  unsigned int *v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  volatile signed __int32 *v22; // rdi
  float *v24; // rdx
  float *v25; // rax
  float *v26; // rdi
  __int64 v27; // rcx
  signed __int32 v28; // eax
  volatile signed __int32 *v29; // rdx
  signed __int32 v30; // ett
  unsigned int v31; // esi
  int v32; // eax
  __int64 *v33; // rdx
  int v34; // eax
  volatile signed __int32 *v35; // rdi
  signed __int32 v36; // eax
  unsigned __int64 v37; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v38; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v39[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 8);
  if ( !v6 )
    goto LABEL_8;
  v8 = v6 - 1;
  v39[0] = (__int64 *)sub_25FD4A0((_QWORD *)a3[1], a3[1] + 4LL * (unsigned int)(*((_DWORD *)a3 + 1) * *(_DWORD *)a3));
  v9 = v8 & sub_35B9BD0((int *)a3, (int *)a3 + 1, (__int64 *)v39);
  v10 = (__int64 *)(v7 + 8LL * v9);
  v11 = *v10;
  if ( !*v10 )
    goto LABEL_8;
  for ( i = 1; ; ++i )
  {
    if ( v11 == 1 || *a3 != *(_QWORD *)(v11 + 24) )
      goto LABEL_6;
    v24 = *(float **)(v11 + 32);
    v25 = (float *)a3[1];
    v26 = &v25[*((_DWORD *)a3 + 1) * *(_DWORD *)a3];
    if ( v25 == v26 )
      break;
    while ( *v25 == *v24 )
    {
      ++v25;
      ++v24;
      if ( v26 == v25 )
        goto LABEL_21;
    }
LABEL_6:
    v9 = v8 & (i + v9);
    v10 = (__int64 *)(v7 + 8LL * v9);
    v11 = *v10;
    if ( !*v10 )
      goto LABEL_8;
  }
LABEL_21:
  if ( v10 != (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
  {
    v22 = *(volatile signed __int32 **)(*v10 + 8);
    v27 = *v10 + 24;
    if ( !v22 )
LABEL_51:
      abort();
    v28 = *((_DWORD *)v22 + 2);
    v29 = v22 + 2;
    do
    {
      if ( !v28 )
        goto LABEL_51;
      v30 = v28;
      v28 = _InterlockedCompareExchange(v29, v28 + 1, v28);
    }
    while ( v30 != v28 );
    *a1 = v27;
    a1[1] = (__int64)v22;
    if ( &_pthread_key_create )
      _InterlockedAdd(v29, 1u);
    else
      ++*((_DWORD *)v22 + 2);
LABEL_16:
    sub_A191D0(v22);
    return a1;
  }
LABEL_8:
  v13 = (_QWORD *)sub_22077B0(0x50u);
  v14 = (volatile signed __int32 *)v13;
  if ( v13 )
  {
    v15 = *a3;
    v13[4] = a2;
    v16 = (__int64)(v13 + 5);
    v17 = (unsigned int *)(v13 + 7);
    *a3 = 0;
    v18 = (unsigned int *)(v13 + 5);
    v13[1] = 0x100000001LL;
    v13[2] = 0;
    *v13 = &unk_4A3A418;
    v19 = a3[1];
    *((_QWORD *)v14 + 3) = 0;
    a3[1] = 0;
    v20 = (unsigned __int64)(v14 + 4);
    *((_QWORD *)v14 + 6) = v19;
    *((_QWORD *)v14 + 5) = v15;
    sub_35B9650(v17, v18);
    v21 = *((_QWORD *)v14 + 3);
    if ( v21 && *(_DWORD *)(v21 + 8) )
      goto LABEL_11;
    *((_QWORD *)v14 + 2) = v20;
    if ( &_pthread_key_create )
      _InterlockedAdd(v14 + 3, 1u);
    else
      ++*((_DWORD *)v14 + 3);
LABEL_37:
    v35 = *(volatile signed __int32 **)(v20 + 8);
    if ( v35 )
    {
      if ( &_pthread_key_create )
      {
        v36 = _InterlockedExchangeAdd(v35 + 3, 0xFFFFFFFF);
      }
      else
      {
        v36 = *((_DWORD *)v35 + 3);
        *((_DWORD *)v35 + 3) = v36 - 1;
      }
      if ( v36 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v35 + 24LL))(v35);
    }
    *(_QWORD *)(v20 + 8) = v14;
  }
  else
  {
    v20 = 16;
    v16 = 40;
    if ( !MEMORY[0x18] || !*(_DWORD *)(MEMORY[0x18] + 8LL) )
    {
      MEMORY[0x10] = 16;
      goto LABEL_37;
    }
  }
LABEL_11:
  v37 = v20;
  if ( !(unsigned __int8)sub_35BDFC0(a2, &v37, &v38) )
  {
    v31 = *(_DWORD *)(a2 + 24);
    v32 = *(_DWORD *)(a2 + 16);
    v33 = v38;
    ++*(_QWORD *)a2;
    v34 = v32 + 1;
    v39[0] = v33;
    if ( 4 * v34 >= 3 * v31 )
    {
      v31 *= 2;
    }
    else if ( v31 - *(_DWORD *)(a2 + 20) - v34 > v31 >> 3 )
    {
LABEL_30:
      *(_DWORD *)(a2 + 16) = v34;
      if ( *v33 )
        --*(_DWORD *)(a2 + 20);
      *v33 = v37;
      goto LABEL_12;
    }
    sub_35BF630(a2, v31);
    sub_35BDFC0(a2, &v37, v39);
    v33 = v39[0];
    v34 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_30;
  }
LABEL_12:
  *a1 = v16;
  a1[1] = (__int64)v14;
  if ( v14 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd(v14 + 2, 1u);
    else
      ++*((_DWORD *)v14 + 2);
    v22 = v14;
    goto LABEL_16;
  }
  return a1;
}
