// Function: sub_23CDA10
// Address: 0x23cda10
//
__int64 __fastcall sub_23CDA10(__int64 a1, __int64 a2)
{
  pthread_cond_t *v2; // r12
  unsigned int v4; // eax
  bool v5; // zf
  __m128i *v6; // rax
  __m128i v7; // xmm1
  __m128i v8; // xmm0
  void (__fastcall *v9)(_QWORD, _QWORD, _QWORD); // rcx
  __int64 v10; // rsi
  int *v11; // rdx
  __m128i v12; // xmm2
  void (__fastcall *v13)(_QWORD, _QWORD, _QWORD); // rax
  pthread_mutex_t *v14; // rdi
  __int64 next; // r15
  int v16; // r14d
  __int64 v17; // r9
  _QWORD *v18; // rdx
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rcx
  int v22; // eax
  void (__fastcall *v23)(pthread_mutex_t *, pthread_mutex_t *, __int64); // rcx
  unsigned int v24; // esi
  __int64 v25; // rcx
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r9
  __int64 result; // rax
  _QWORD *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r15
  int v33; // eax
  int v34; // eax
  int v35; // eax
  int v36; // r14d
  int v37; // r14d
  __int64 v38; // r11
  __int64 v39; // rcx
  __int64 v40; // r9
  int v41; // edi
  int v42; // r14d
  int v43; // r14d
  int v44; // edi
  __int64 v45; // r11
  __int64 v46; // rcx
  __int64 v47; // r9
  int v48; // r8d
  unsigned int v49; // [rsp+4h] [rbp-9Ch]
  pthread_mutex_t *mutex; // [rsp+18h] [rbp-88h]
  pthread_mutex_t *v52; // [rsp+20h] [rbp-80h] BYREF
  char v53; // [rsp+28h] [rbp-78h]
  __m128i v54; // [rsp+30h] [rbp-70h] BYREF
  void (__fastcall *v55)(_QWORD, _QWORD, _QWORD); // [rsp+40h] [rbp-60h]
  int *v56; // [rsp+48h] [rbp-58h]
  __m128i v57; // [rsp+50h] [rbp-50h] BYREF
  void (__fastcall *v58)(_QWORD, _QWORD, _QWORD); // [rsp+60h] [rbp-40h]
  int *v59; // [rsp+68h] [rbp-38h]

  v2 = (pthread_cond_t *)(a1 + 216);
  mutex = (pthread_mutex_t *)(a1 + 176);
  while ( 1 )
  {
    v55 = 0;
    v53 = 0;
    v52 = mutex;
    if ( &_pthread_key_create )
    {
      v4 = pthread_mutex_lock(mutex);
      if ( v4 )
LABEL_92:
        sub_4264C5(v4);
    }
    v5 = *(_BYTE *)(a1 + 352) == 0;
    v53 = 1;
    if ( v5 )
      break;
    while ( *(_QWORD *)(a1 + 144) == *(_QWORD *)(a1 + 112) )
    {
      if ( a2 && (unsigned __int8)sub_23CCCA0(a1, a2) )
        goto LABEL_38;
      sub_2210B30(v2, &v52);
      if ( !*(_BYTE *)(a1 + 352) )
        goto LABEL_5;
    }
    if ( !*(_BYTE *)(a1 + 352) )
      break;
LABEL_6:
    v6 = *(__m128i **)(a1 + 112);
    v7 = _mm_loadu_si128(&v57);
    ++*(_DWORD *)(a1 + 312);
    v8 = _mm_loadu_si128(v6);
    v9 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v6[1].m128i_i64[0];
    *v6 = v7;
    v6[1].m128i_i64[0] = 0;
    v10 = (__int64)v59;
    v11 = (int *)v6[1].m128i_i64[1];
    v12 = _mm_loadu_si128(&v54);
    v54 = v8;
    v6[1].m128i_i64[1] = (__int64)v59;
    v13 = v55;
    v55 = v9;
    v58 = v13;
    v59 = v56;
    v56 = v11;
    v57 = v12;
    if ( v13 )
    {
      v10 = (__int64)&v57;
      v13(&v57, &v57, 3);
    }
    v14 = *(pthread_mutex_t **)(a1 + 112);
    next = (__int64)v14->__list.__next;
    if ( next )
    {
      v10 = *(unsigned int *)(a1 + 344);
      if ( (_DWORD)v10 )
      {
        v16 = 1;
        v17 = *(_QWORD *)(a1 + 328);
        v18 = 0;
        v19 = (v10 - 1) & (((unsigned int)next >> 9) ^ ((unsigned int)next >> 4));
        v20 = (__int64 *)(v17 + 16LL * v19);
        v21 = *v20;
        if ( next == *v20 )
        {
LABEL_11:
          v11 = (int *)(v20 + 1);
          v22 = *((_DWORD *)v20 + 2) + 1;
LABEL_12:
          *v11 = v22;
          v14 = *(pthread_mutex_t **)(a1 + 112);
          goto LABEL_13;
        }
        while ( v21 != -4096 )
        {
          if ( v21 == -8192 && !v18 )
            v18 = v20;
          v19 = (v10 - 1) & (v16 + v19);
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( next == *v20 )
            goto LABEL_11;
          ++v16;
        }
        if ( !v18 )
          v18 = v20;
        v34 = *(_DWORD *)(a1 + 336);
        ++*(_QWORD *)(a1 + 320);
        v35 = v34 + 1;
        if ( 4 * v35 < (unsigned int)(3 * v10) )
        {
          if ( (int)v10 - *(_DWORD *)(a1 + 340) - v35 <= (unsigned int)v10 >> 3 )
          {
            v49 = ((unsigned int)next >> 9) ^ ((unsigned int)next >> 4);
            sub_23CD830(a1 + 320, v10);
            v42 = *(_DWORD *)(a1 + 344);
            if ( !v42 )
            {
LABEL_99:
              ++*(_DWORD *)(a1 + 336);
              BUG();
            }
            v43 = v42 - 1;
            v44 = 1;
            v10 = 0;
            v45 = *(_QWORD *)(a1 + 328);
            LODWORD(v46) = v43 & v49;
            v35 = *(_DWORD *)(a1 + 336) + 1;
            v18 = (_QWORD *)(v45 + 16LL * (v43 & v49));
            v47 = *v18;
            if ( next != *v18 )
            {
              while ( v47 != -4096 )
              {
                if ( !v10 && v47 == -8192 )
                  v10 = (__int64)v18;
                v46 = v43 & (unsigned int)(v46 + v44);
                v18 = (_QWORD *)(v45 + 16 * v46);
                v47 = *v18;
                if ( next == *v18 )
                  goto LABEL_68;
                ++v44;
              }
              goto LABEL_76;
            }
          }
          goto LABEL_68;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 320);
      }
      v10 = (unsigned int)(2 * v10);
      sub_23CD830(a1 + 320, v10);
      v36 = *(_DWORD *)(a1 + 344);
      if ( !v36 )
        goto LABEL_99;
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 328);
      LODWORD(v39) = v37 & (((unsigned int)next >> 9) ^ ((unsigned int)next >> 4));
      v35 = *(_DWORD *)(a1 + 336) + 1;
      v18 = (_QWORD *)(v38 + 16LL * (unsigned int)v39);
      v40 = *v18;
      if ( next != *v18 )
      {
        v41 = 1;
        v10 = 0;
        while ( v40 != -4096 )
        {
          if ( v40 == -8192 && !v10 )
            v10 = (__int64)v18;
          v39 = v37 & (unsigned int)(v39 + v41);
          v18 = (_QWORD *)(v38 + 16 * v39);
          v40 = *v18;
          if ( next == *v18 )
            goto LABEL_68;
          ++v41;
        }
LABEL_76:
        if ( v10 )
          v18 = (_QWORD *)v10;
      }
LABEL_68:
      *(_DWORD *)(a1 + 336) = v35;
      if ( *v18 != -4096 )
        --*(_DWORD *)(a1 + 340);
      *v18 = next;
      v22 = 1;
      v11 = (int *)(v18 + 1);
      *v11 = 0;
      goto LABEL_12;
    }
LABEL_13:
    v23 = (void (__fastcall *)(pthread_mutex_t *, pthread_mutex_t *, __int64))*(&v14->__align + 2);
    if ( v14 == (pthread_mutex_t *)(*(_QWORD *)(a1 + 128) - 40LL) )
    {
      if ( v23 )
        v23(v14, v14, 3);
      v14 = *(pthread_mutex_t **)(a1 + 120);
      v10 = 480;
      j_j___libc_free_0((unsigned __int64)v14);
      v30 = (_QWORD *)(*(_QWORD *)(a1 + 136) + 8LL);
      *(_QWORD *)(a1 + 136) = v30;
      v31 = *v30;
      v11 = (int *)(*v30 + 480LL);
      *(_QWORD *)(a1 + 120) = v31;
      *(_QWORD *)(a1 + 128) = v11;
      *(_QWORD *)(a1 + 112) = v31;
    }
    else
    {
      if ( v23 )
      {
        v10 = (__int64)v14;
        v23(v14, v14, 3);
      }
      *(_QWORD *)(a1 + 112) += 40LL;
    }
    if ( v53 )
    {
      v14 = v52;
      if ( v52 )
      {
        if ( &_pthread_key_create )
          pthread_mutex_unlock(v52);
      }
    }
    if ( !v55 )
      sub_4263D6(v14, v10, v11);
    ((void (__fastcall *)(__m128i *))v56)(&v54);
    if ( &_pthread_key_create )
    {
      v4 = pthread_mutex_lock(mutex);
      if ( v4 )
        goto LABEL_92;
    }
    --*(_DWORD *)(a1 + 312);
    if ( next )
    {
      v24 = *(_DWORD *)(a1 + 344);
      v25 = *(_QWORD *)(a1 + 328);
      if ( v24 )
      {
        v26 = (v24 - 1) & (((unsigned int)next >> 9) ^ ((unsigned int)next >> 4));
        v27 = (__int64 *)(v25 + 16LL * v26);
        v28 = *v27;
        if ( next == *v27 )
        {
LABEL_24:
          v5 = (*((_DWORD *)v27 + 2))-- == 1;
          if ( v5 )
          {
            *v27 = -8192;
            --*(_DWORD *)(a1 + 336);
            ++*(_DWORD *)(a1 + 340);
          }
          if ( (unsigned __int8)sub_23CCCA0(a1, next) )
          {
            if ( &_pthread_key_create )
              pthread_mutex_unlock(mutex);
            sub_2210B70((pthread_cond_t *)(a1 + 264));
            sub_2210B70(v2);
          }
          else if ( &_pthread_key_create )
          {
            pthread_mutex_unlock(mutex);
          }
          goto LABEL_30;
        }
        v33 = 1;
        while ( v28 != -4096 )
        {
          v48 = v33 + 1;
          v26 = (v24 - 1) & (v33 + v26);
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( next == *v27 )
            goto LABEL_24;
          v33 = v48;
        }
      }
      v27 = (__int64 *)(v25 + 16LL * v24);
      goto LABEL_24;
    }
    v32 = (unsigned __int8)sub_23CCCA0(a1, 0);
    if ( &_pthread_key_create )
      pthread_mutex_unlock(mutex);
    if ( (_BYTE)v32 )
      sub_2210B70((pthread_cond_t *)(a1 + 264));
LABEL_30:
    if ( v55 )
      v55(&v54, &v54, 3);
  }
LABEL_5:
  if ( *(_QWORD *)(a1 + 112) != *(_QWORD *)(a1 + 144) )
    goto LABEL_6;
LABEL_38:
  if ( v53 && v52 && &_pthread_key_create )
    pthread_mutex_unlock(v52);
  result = (__int64)v55;
  if ( v55 )
    return ((__int64 (__fastcall *)(__m128i *, __m128i *, __int64))v55)(&v54, &v54, 3);
  return result;
}
