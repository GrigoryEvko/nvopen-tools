// Function: sub_1ECFE60
// Address: 0x1ecfe60
//
__int64 *__fastcall sub_1ECFE60(__int64 *a1, __int64 a2, unsigned int *a3)
{
  int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // r14
  int v9; // edx
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  volatile signed __int32 *v12; // rdi
  signed __int32 v13; // eax
  char v14; // al
  __int64 *v15; // rdx
  __int64 v16; // rdi
  int v17; // r14d
  unsigned int v18; // eax
  __int64 v19; // r9
  unsigned int v20; // r8d
  __int64 *v21; // rcx
  __int64 v22; // rax
  int i; // r10d
  __int64 v24; // rdx
  size_t v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  signed __int32 v28; // eax
  volatile signed __int32 *v29; // rdx
  signed __int32 v30; // ett
  unsigned int v32; // esi
  int v33; // eax
  int v34; // eax
  int v35; // eax
  __int64 *v36; // [rsp+8h] [rbp-58h]
  int v37; // [rsp+10h] [rbp-50h]
  unsigned int v38; // [rsp+14h] [rbp-4Ch]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+20h] [rbp-40h] BYREF
  __int64 *v42[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a2 + 24);
  v39 = *(_QWORD *)(a2 + 8);
  if ( v6 )
  {
    v17 = v6 - 1;
    v42[0] = (__int64 *)sub_1ECD5B0(*((_QWORD **)a3 + 1), *((_QWORD *)a3 + 1) + 4LL * *a3);
    v18 = sub_18FDAA0((int *)a3, (__int64 *)v42);
    v19 = v39;
    v20 = v17 & v18;
    v21 = (__int64 *)(v39 + 8LL * (v17 & v18));
    v22 = *v21;
    if ( *v21 )
    {
      for ( i = 1; ; ++i )
      {
        if ( v22 != 1 )
        {
          v24 = *a3;
          if ( (_DWORD)v24 == *(_DWORD *)(v22 + 24) )
          {
            v25 = 4 * v24;
            if ( !v25 )
            {
              v26 = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
              goto LABEL_22;
            }
            v37 = i;
            v38 = v20;
            v40 = v19;
            v36 = v21;
            v35 = memcmp(*((const void **)a3 + 1), *(const void **)(v22 + 32), v25);
            v19 = v40;
            v20 = v38;
            i = v37;
            if ( !v35 )
              break;
          }
        }
        v20 = v17 & (i + v20);
        v21 = (__int64 *)(v19 + 8LL * v20);
        v22 = *v21;
        if ( !*v21 )
          goto LABEL_2;
      }
      v21 = v36;
      v26 = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
LABEL_22:
      if ( v21 != (__int64 *)v26 )
      {
        v16 = *(_QWORD *)(*v21 + 8);
        v27 = *v21 + 24;
        if ( !v16 )
LABEL_50:
          abort();
        v28 = *(_DWORD *)(v16 + 8);
        v29 = (volatile signed __int32 *)(v16 + 8);
        do
        {
          if ( !v28 )
            goto LABEL_50;
          v30 = v28;
          v28 = _InterlockedCompareExchange(v29, v28 + 1, v28);
        }
        while ( v30 != v28 );
        *a1 = v27;
        a1[1] = v16;
        if ( &_pthread_key_create )
          _InterlockedAdd(v29, 1u);
        else
          ++*(_DWORD *)(v16 + 8);
        goto LABEL_29;
      }
    }
  }
LABEL_2:
  v7 = sub_22077B0(56);
  v8 = v7;
  if ( v7 )
  {
    v9 = *a3;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 8) = 0x100000001LL;
    *(_QWORD *)(v7 + 32) = a2;
    *(_DWORD *)(v7 + 40) = v9;
    *(_QWORD *)v7 = &unk_49FDE98;
    v10 = *((_QWORD *)a3 + 1);
    *((_QWORD *)a3 + 1) = 0;
    v11 = v8 + 16;
    *(_QWORD *)(v8 + 48) = v10;
    *(_QWORD *)(v8 + 16) = v8 + 16;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(v8 + 12), 1u);
    else
      *(_DWORD *)(v8 + 12) = 2;
  }
  else
  {
    v11 = 16;
    if ( MEMORY[0x18] && *(_DWORD *)(MEMORY[0x18] + 8LL) )
      goto LABEL_11;
    MEMORY[0x10] = 16;
  }
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
LABEL_11:
  v41 = v11;
  v14 = sub_1ECEDE0(a2, &v41, v42);
  v15 = v42[0];
  if ( v14 )
    goto LABEL_12;
  v32 = *(_DWORD *)(a2 + 24);
  v33 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v34 = v33 + 1;
  if ( 4 * v34 >= 3 * v32 )
  {
    v32 *= 2;
LABEL_46:
    sub_1ECFD00(a2, v32);
    sub_1ECEDE0(a2, &v41, v42);
    v15 = v42[0];
    v34 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_33;
  }
  if ( v32 - *(_DWORD *)(a2 + 20) - v34 <= v32 >> 3 )
    goto LABEL_46;
LABEL_33:
  *(_DWORD *)(a2 + 16) = v34;
  if ( *v15 )
    --*(_DWORD *)(a2 + 20);
  *v15 = v41;
LABEL_12:
  a1[1] = v8;
  *a1 = v11 + 24;
  if ( v8 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(v8 + 8), 1u);
    else
      ++*(_DWORD *)(v8 + 8);
    v16 = v8;
LABEL_29:
    sub_A191D0((volatile signed __int32 *)v16);
  }
  return a1;
}
