// Function: sub_1ED17B0
// Address: 0x1ed17b0
//
__int64 *__fastcall sub_1ED17B0(__int64 *a1, __int64 a2, __int64 *a3)
{
  int v6; // r15d
  _QWORD *v7; // rax
  volatile signed __int32 *v8; // r15
  __int64 v9; // rdx
  unsigned int *v10; // rsi
  unsigned int *v11; // rdi
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  char v15; // al
  __int64 *v16; // rdx
  volatile signed __int32 *v17; // rdi
  int v19; // r15d
  unsigned int v20; // edi
  __int64 *v21; // rsi
  __int64 v22; // rax
  int i; // r9d
  float *v24; // rdx
  float *v25; // rax
  float *v26; // r8
  unsigned int v27; // esi
  int v28; // eax
  int v29; // eax
  __int64 v30; // rcx
  signed __int32 v31; // eax
  volatile signed __int32 *v32; // rdx
  signed __int32 v33; // ett
  volatile signed __int32 *v34; // rdi
  signed __int32 v35; // eax
  __int64 v36; // [rsp+8h] [rbp-48h]
  __int64 v37; // [rsp+8h] [rbp-48h]
  unsigned __int64 v38; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v39[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a2 + 24);
  v36 = *(_QWORD *)(a2 + 8);
  if ( !v6 )
    goto LABEL_2;
  v19 = v6 - 1;
  v39[0] = (__int64 *)sub_1ECD5B0((_QWORD *)a3[1], a3[1] + 4LL * (unsigned int)(*((_DWORD *)a3 + 1) * *(_DWORD *)a3));
  v20 = v19 & sub_1ECC960(a3, (int *)a3 + 1, v39);
  v21 = (__int64 *)(v36 + 8LL * v20);
  v22 = *v21;
  if ( !*v21 )
    goto LABEL_2;
  for ( i = 1; ; ++i )
  {
    if ( v22 == 1 || *a3 != *(_QWORD *)(v22 + 24) )
      goto LABEL_20;
    v24 = *(float **)(v22 + 32);
    v25 = (float *)a3[1];
    v26 = &v25[*((_DWORD *)a3 + 1) * *(_DWORD *)a3];
    if ( v25 == v26 )
      break;
    while ( *v25 == *v24 )
    {
      ++v25;
      ++v24;
      if ( v26 == v25 )
        goto LABEL_27;
    }
LABEL_20:
    v20 = v19 & (i + v20);
    v21 = (__int64 *)(v36 + 8LL * v20);
    v22 = *v21;
    if ( !*v21 )
      goto LABEL_2;
  }
LABEL_27:
  if ( (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) != v21 )
  {
    v17 = *(volatile signed __int32 **)(*v21 + 8);
    v30 = *v21 + 24;
    if ( !v17 )
LABEL_52:
      abort();
    v31 = *((_DWORD *)v17 + 2);
    v32 = v17 + 2;
    do
    {
      if ( !v31 )
        goto LABEL_52;
      v33 = v31;
      v31 = _InterlockedCompareExchange(v32, v31 + 1, v31);
    }
    while ( v33 != v31 );
    *a1 = v30;
    a1[1] = (__int64)v17;
    if ( &_pthread_key_create )
      _InterlockedAdd(v32, 1u);
    else
      ++*((_DWORD *)v17 + 2);
LABEL_10:
    sub_A191D0(v17);
    return a1;
  }
LABEL_2:
  v7 = (_QWORD *)sub_22077B0(80);
  v8 = (volatile signed __int32 *)v7;
  if ( v7 )
  {
    v9 = *a3;
    v7[4] = a2;
    v10 = (unsigned int *)(v7 + 5);
    v11 = (unsigned int *)(v7 + 7);
    *a3 = 0;
    v7[1] = 0x100000001LL;
    v7[2] = 0;
    *v7 = &unk_49FDF08;
    v12 = a3[1];
    *((_QWORD *)v8 + 3) = 0;
    a3[1] = 0;
    v13 = (unsigned __int64)(v8 + 4);
    *((_QWORD *)v8 + 6) = v12;
    *((_QWORD *)v8 + 5) = v9;
    v37 = (__int64)(v8 + 10);
    sub_1ECBD10(v11, v10);
    v14 = *((_QWORD *)v8 + 3);
    if ( v14 && *(_DWORD *)(v14 + 8) )
      goto LABEL_5;
    *((_QWORD *)v8 + 2) = v13;
    if ( &_pthread_key_create )
      _InterlockedAdd(v8 + 3, 1u);
    else
      ++*((_DWORD *)v8 + 3);
LABEL_37:
    v34 = *(volatile signed __int32 **)(v13 + 8);
    if ( v34 )
    {
      if ( &_pthread_key_create )
      {
        v35 = _InterlockedExchangeAdd(v34 + 3, 0xFFFFFFFF);
      }
      else
      {
        v35 = *((_DWORD *)v34 + 3);
        *((_DWORD *)v34 + 3) = v35 - 1;
      }
      if ( v35 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v34 + 24LL))(v34);
    }
    *(_QWORD *)(v13 + 8) = v8;
  }
  else
  {
    v13 = 16;
    v37 = 40;
    if ( !MEMORY[0x18] || !*(_DWORD *)(MEMORY[0x18] + 8LL) )
    {
      MEMORY[0x10] = 16;
      goto LABEL_37;
    }
  }
LABEL_5:
  v38 = v13;
  v15 = sub_1ED03D0(a2, &v38, v39);
  v16 = v39[0];
  if ( !v15 )
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
LABEL_24:
      *(_DWORD *)(a2 + 16) = v29;
      if ( *v16 )
        --*(_DWORD *)(a2 + 20);
      *v16 = v38;
      goto LABEL_6;
    }
    sub_1ED1650(a2, v27);
    sub_1ED03D0(a2, &v38, v39);
    v16 = v39[0];
    v29 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_24;
  }
LABEL_6:
  a1[1] = (__int64)v8;
  *a1 = v37;
  if ( v8 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd(v8 + 2, 1u);
    else
      ++*((_DWORD *)v8 + 2);
    v17 = v8;
    goto LABEL_10;
  }
  return a1;
}
