// Function: sub_16D5230
// Address: 0x16d5230
//
__int64 *__fastcall sub_16D5230(__int64 *a1, __int64 a2, __m128i *a3)
{
  volatile signed __int32 *v5; // r12
  __int64 v6; // r14
  volatile signed __int32 *v7; // rax
  char v8; // al
  int (**v9)(pthread_key_t *, void (*)(void *)); // rcx
  signed __int32 v10; // eax
  pthread_mutex_t *v11; // r8
  unsigned int v12; // eax
  unsigned __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  signed __int32 v18; // eax
  char *v19; // rdx
  char *v20; // r11
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rdx
  volatile signed __int32 *v25; // rdx
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r10
  __int64 v29; // rax
  __int64 v30; // rax
  const void *v31; // rsi
  pthread_mutex_t *v32; // r8
  void *v33; // r10
  __int64 v34; // rdx
  void *v35; // rax
  char *v36; // r10
  __int64 v37; // rax
  __int64 v38; // rax
  char *v39; // rdx
  void *v40; // rdi
  char *v41; // rax
  pthread_mutex_t *v42; // [rsp+8h] [rbp-78h]
  pthread_mutex_t *v43; // [rsp+8h] [rbp-78h]
  char *v44; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+18h] [rbp-68h]
  pthread_mutex_t *v46; // [rsp+18h] [rbp-68h]
  pthread_mutex_t *v47; // [rsp+20h] [rbp-60h]
  unsigned __int64 v48; // [rsp+20h] [rbp-60h]
  pthread_mutex_t *v49; // [rsp+20h] [rbp-60h]
  char *v50; // [rsp+20h] [rbp-60h]
  __int64 v51; // [rsp+28h] [rbp-58h]
  char *v52; // [rsp+28h] [rbp-58h]
  __int64 v53; // [rsp+40h] [rbp-40h] BYREF
  volatile signed __int32 *v54; // [rsp+48h] [rbp-38h]

  sub_16D42F0(&v53, a3);
  v5 = v54;
  v6 = v53;
  if ( v54 )
  {
    v7 = v54 + 2;
    if ( &_pthread_key_create )
    {
      _InterlockedAdd(v7, 1u);
      _InterlockedAdd(v7, 1u);
    }
    else
    {
      ++*((_DWORD *)v54 + 2);
      ++*((_DWORD *)v5 + 2);
    }
  }
  if ( !v6 )
    sub_42641C(3u);
  v8 = *(_BYTE *)(v6 + 20);
  *(_BYTE *)(v6 + 20) = 1;
  if ( v8 )
    sub_42641C(1u);
  v9 = &_pthread_key_create;
  if ( v5 )
  {
    if ( &_pthread_key_create )
    {
      v10 = _InterlockedExchangeAdd(v5 + 2, 0xFFFFFFFF);
    }
    else
    {
      v10 = *((_DWORD *)v5 + 2);
      *((_DWORD *)v5 + 2) = v10 - 1;
    }
    if ( v10 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v5 + 16LL))(v5);
      v9 = &_pthread_key_create;
      if ( &_pthread_key_create )
      {
        v18 = _InterlockedExchangeAdd(v5 + 3, 0xFFFFFFFF);
      }
      else
      {
        v18 = *((_DWORD *)v5 + 3);
        *((_DWORD *)v5 + 3) = v18 - 1;
      }
      if ( v18 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v5 + 24LL))(v5);
        v9 = &_pthread_key_create;
      }
    }
  }
  v11 = (pthread_mutex_t *)(a2 + 104);
  if ( &_pthread_key_create )
  {
    v12 = pthread_mutex_lock((pthread_mutex_t *)(a2 + 104));
    v11 = (pthread_mutex_t *)(a2 + 104);
    v9 = &_pthread_key_create;
    if ( v12 )
      sub_4264C5(v12);
  }
  v13 = *(_QWORD *)(a2 + 88);
  v14 = *(__int64 **)(a2 + 72);
  v15 = v13 - 16;
  if ( v14 == (__int64 *)(v13 - 16) )
  {
    v19 = *(char **)(a2 + 96);
    v20 = *(char **)(a2 + 64);
    v51 = v19 - v20;
    if ( ((__int64)(*(_QWORD *)(a2 + 56) - *(_QWORD *)(a2 + 40)) >> 4)
       + 32 * (((v19 - v20) >> 3) - 1)
       + (((__int64)v14 - *(_QWORD *)(a2 + 80)) >> 4) == 0x7FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v21 = *(_QWORD *)(a2 + 24);
    v13 = *(_QWORD *)(a2 + 32);
    if ( v13 - ((__int64)&v19[-v21] >> 3) <= 1 )
    {
      v28 = (v51 >> 3) + 2;
      if ( v13 > 2 * v28 )
      {
        v39 = v19 + 8;
        v13 = (v13 - v28) >> 1;
        v36 = (char *)(v21 + 8 * v13);
        if ( v20 <= v36 )
        {
          if ( v20 != v39 )
          {
            v13 = *(_QWORD *)(a2 + 64);
            v46 = v11;
            v50 = v36;
            memmove(&v36[*(_QWORD *)(a2 + 96) + 8LL - (_QWORD)v39], v20, v39 - v20);
            v36 = v50;
            v11 = v46;
          }
        }
        else if ( v20 != v39 )
        {
          v40 = (void *)(v21 + 8 * v13);
          v13 = *(_QWORD *)(a2 + 64);
          v49 = v11;
          v41 = (char *)memmove(v40, v20, v39 - v20);
          v11 = v49;
          v36 = v41;
        }
      }
      else
      {
        v29 = 1;
        if ( v13 )
          v29 = *(_QWORD *)(a2 + 32);
        v48 = v13 + v29 + 2;
        if ( v48 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v21, v48, v19);
        v42 = v11;
        v30 = sub_22077B0(8 * v48);
        v31 = *(const void **)(a2 + 64);
        v45 = v30;
        v32 = v42;
        v33 = (void *)(v30 + 8 * ((v48 - ((v51 >> 3) + 2)) >> 1));
        v34 = *(_QWORD *)(a2 + 96) + 8LL;
        if ( (const void *)v34 != v31 )
        {
          v35 = memmove(v33, v31, v34 - (_QWORD)v31);
          v32 = v42;
          v33 = v35;
        }
        v43 = v32;
        v13 = 8LL * *(_QWORD *)(a2 + 32);
        v44 = (char *)v33;
        j_j___libc_free_0(*(_QWORD *)(a2 + 24), v13);
        v11 = v43;
        v36 = v44;
        *(_QWORD *)(a2 + 24) = v45;
        *(_QWORD *)(a2 + 32) = v48;
      }
      *(_QWORD *)(a2 + 64) = v36;
      v37 = *(_QWORD *)v36;
      v19 = &v36[v51];
      *(_QWORD *)(a2 + 48) = *(_QWORD *)v36;
      *(_QWORD *)(a2 + 56) = v37 + 512;
      *(_QWORD *)(a2 + 96) = &v36[v51];
      v38 = *(_QWORD *)&v36[v51];
      *(_QWORD *)(a2 + 80) = v38;
      *(_QWORD *)(a2 + 88) = v38 + 512;
    }
    v47 = v11;
    v52 = v19;
    v22 = sub_22077B0(512);
    v11 = v47;
    v9 = &_pthread_key_create;
    *((_QWORD *)v52 + 1) = v22;
    v23 = *(__int64 **)(a2 + 72);
    if ( v23 )
    {
      *v23 = 0;
      v24 = v53;
      v23[1] = 0;
      *v23 = v24;
      v25 = v54;
      v53 = 0;
      v54 = 0;
      v23[1] = (__int64)v25;
    }
    v26 = (__int64 *)(*(_QWORD *)(a2 + 96) + 8LL);
    *(_QWORD *)(a2 + 96) = v26;
    v27 = *v26;
    v15 = *v26 + 512;
    *(_QWORD *)(a2 + 80) = v27;
    *(_QWORD *)(a2 + 88) = v15;
    *(_QWORD *)(a2 + 72) = v27;
  }
  else
  {
    if ( v14 )
    {
      *v14 = 0;
      v16 = v53;
      v14[1] = 0;
      *v14 = v16;
      v15 = (__int64)v54;
      v53 = 0;
      v54 = 0;
      v14[1] = v15;
      v14 = *(__int64 **)(a2 + 72);
    }
    *(_QWORD *)(a2 + 72) = v14 + 2;
  }
  if ( &_pthread_key_create )
    pthread_mutex_unlock(v11);
  sub_2210B50(a2 + 144, v13, v15, v9, v11);
  *a1 = v6;
  a1[1] = (__int64)v5;
  sub_16D4F80(&v53);
  return a1;
}
