// Function: sub_2265950
// Address: 0x2265950
//
void __fastcall sub_2265950(__int64 a1, unsigned int a2, __int64 a3, __int64 *a4, __m128i a5)
{
  __int64 v8; // rdi
  _DWORD *v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // r9
  int v12; // r11d
  _QWORD *v13; // rax
  _QWORD **i; // rsi
  bool v15; // zf
  _QWORD *v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rdx
  _BYTE *v20; // rsi
  _QWORD *v21; // rax
  char v22; // dl
  _QWORD *v23; // rax
  _QWORD **v24; // rdx
  __int64 v25; // rdi
  _QWORD **v26; // r12
  _BYTE *v27; // rbx
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rdi
  _DWORD *v30; // rax
  _QWORD *v31; // rdx
  _QWORD **v32; // r12
  _BYTE *v33; // rbx
  unsigned __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rdi
  __int64 v37; // [rsp-10h] [rbp-290h]
  __int64 v38; // [rsp+8h] [rbp-278h]
  __int64 v39; // [rsp+28h] [rbp-258h]
  _QWORD *v40; // [rsp+30h] [rbp-250h] BYREF
  _QWORD **v41; // [rsp+38h] [rbp-248h] BYREF
  const __m128i *v42[4]; // [rsp+40h] [rbp-240h] BYREF
  char *v43; // [rsp+60h] [rbp-220h] BYREF
  const __m128i *v44; // [rsp+68h] [rbp-218h]
  _QWORD v45[14]; // [rsp+70h] [rbp-210h] BYREF
  _QWORD v46[2]; // [rsp+E0h] [rbp-1A0h] BYREF
  _QWORD *v47; // [rsp+F0h] [rbp-190h]
  __int64 v48; // [rsp+F8h] [rbp-188h]
  _QWORD v49[3]; // [rsp+100h] [rbp-180h] BYREF
  int v50; // [rsp+118h] [rbp-168h]
  _QWORD *v51; // [rsp+120h] [rbp-160h]
  __int64 v52; // [rsp+128h] [rbp-158h]
  _BYTE v53[16]; // [rsp+130h] [rbp-150h] BYREF
  _QWORD *v54; // [rsp+140h] [rbp-140h]
  __int64 v55; // [rsp+148h] [rbp-138h]
  _BYTE v56[16]; // [rsp+150h] [rbp-130h] BYREF
  unsigned __int64 v57; // [rsp+160h] [rbp-120h]
  __int64 v58; // [rsp+168h] [rbp-118h]
  __int64 v59; // [rsp+170h] [rbp-110h]
  _BYTE *v60; // [rsp+178h] [rbp-108h]
  __int64 v61; // [rsp+180h] [rbp-100h]
  _BYTE v62[248]; // [rsp+188h] [rbp-F8h] BYREF

  v8 = **(_QWORD **)a1;
  if ( v8 && (unsigned int)sub_16827A0(v8) )
    goto LABEL_74;
  v9 = (_DWORD *)sub_CEECD0(4, 4u);
  *v9 = 2;
  sub_C94E10((__int64)qword_4F86310, v9);
  sub_C7DA90(&v40, *a4, a4[1], byte_3F871B3, 0, 0);
  v47 = v49;
  v54 = v56;
  v61 = 0x400000000LL;
  memset(&v45[2], 0, 0x58u);
  v10 = *(_QWORD **)(a1 + 8);
  v46[0] = 0;
  v46[1] = 0;
  v48 = 0;
  LOBYTE(v49[0]) = 0;
  v49[2] = 0;
  v50 = 0;
  v51 = v53;
  v52 = 0;
  v53[0] = 0;
  v55 = 0;
  v56[0] = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = v62;
  v38 = a2;
  v39 = *(_QWORD *)(*v10 + 8LL * a2);
  sub_C7EC60(v42, v40);
  sub_E46810(
    (unsigned __int64 *)&v41,
    (__int64)v46,
    v39,
    (__int64)&v43,
    a5,
    (__int64)v46,
    v11,
    v42[0],
    (unsigned __int64)v42[1],
    (__int64)v42[2]->m128i_i64,
    (__int64)v42[3]->m128i_i64);
  if ( LOBYTE(v45[12]) )
  {
    LOBYTE(v45[12]) = 0;
    if ( v45[10] )
      ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v45[10])(&v45[8], &v45[8], 3);
  }
  if ( LOBYTE(v45[7]) )
  {
    LOBYTE(v45[7]) = 0;
    if ( v45[5] )
      ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v45[5])(&v45[3], &v45[3], 3);
  }
  if ( LOBYTE(v45[2]) )
  {
    LOBYTE(v45[2]) = 0;
    if ( v45[0] )
      ((void (__fastcall *)(char **, char **, __int64))v45[0])(&v43, &v43, 3);
  }
  v12 = (int)v41;
  v13 = v41[4];
  for ( i = v41 + 3; v13 != i; v13 = (_QWORD *)v13[1] )
  {
    if ( !v13 )
    {
      MEMORY[0x20] &= 0xFFFFFFF0;
      BUG();
    }
    v15 = (*(_BYTE *)(v13 - 3) & 0x30) == 0;
    *((_BYTE *)v13 - 24) &= 0xF0u;
    if ( !v15 )
      *((_BYTE *)v13 - 23) |= 0x40u;
  }
  sub_226C400(
    **(_DWORD **)(a1 + 16),
    **(_QWORD **)(a1 + 24),
    v12,
    a3,
    *(_QWORD *)(a1 + 32),
    **(_QWORD **)(a1 + 40),
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL));
  v16 = *(_QWORD **)(a1 + 40);
  v17 = v37;
  if ( !*v16 || (v18 = 0, !((unsigned int (__fastcall *)(_QWORD, _QWORD))*v16)(v16[1], 0)) )
  {
    v20 = v41 + 1;
    v21 = v41[2];
    if ( v41 + 1 != v21 )
    {
      do
      {
        if ( !v21 )
          BUG();
        v17 = *(_BYTE *)(v21 - 3) & 0xF;
        if ( (_BYTE)v17 != 6 )
        {
          if ( (_BYTE)v17 )
          {
            v22 = *(_BYTE *)(v21 - 3) & 0xF0 | 3;
            *((_BYTE *)v21 - 24) = v22;
            if ( (v22 & 0x30) != 0 )
              *((_BYTE *)v21 - 23) |= 0x40u;
          }
        }
        v21 = (_QWORD *)v21[1];
      }
      while ( v20 != (_BYTE *)v21 );
    }
    if ( *(int *)(a3 + 1560) < 0 )
    {
LABEL_21:
      v23 = *(_QWORD **)(a1 + 88);
      v24 = v41;
      v41 = 0;
      *(_QWORD *)(*v23 + 8 * v38) = v24;
      v25 = **(_QWORD **)a1;
      if ( !v25 || !(unsigned int)sub_1682970(v25) )
      {
        v26 = v41;
        if ( v41 )
        {
          sub_BA9C10(v41, (__int64)v20, (__int64)v24, v17);
          j_j___libc_free_0((unsigned __int64)v26);
        }
        v27 = v60;
        v28 = (unsigned __int64)&v60[48 * (unsigned int)v61];
        if ( v60 != (_BYTE *)v28 )
        {
          do
          {
            v28 -= 48LL;
            v29 = *(_QWORD *)(v28 + 16);
            if ( v29 != v28 + 32 )
              j_j___libc_free_0(v29);
          }
          while ( v27 != (_BYTE *)v28 );
LABEL_29:
          v28 = (unsigned __int64)v60;
          goto LABEL_30;
        }
        goto LABEL_30;
      }
LABEL_74:
      sub_C64ED0("GNU Jobserver support requested, but an error occurred", 1u);
    }
    v44 = 0;
    v43 = (char *)v45;
    LOBYTE(v45[0]) = 0;
    v30 = (_DWORD *)sub_CEECD0(4, 4u);
    *v30 = 4;
    sub_C94E10((__int64)qword_4F86310, v30);
    if ( !(unsigned __int8)sub_3099970(*(_QWORD *)(a1 + 48), v41, &v43, 0, *(_QWORD *)(a1 + 40)) )
      **(_BYTE **)(a1 + 56) = 0;
    v31 = *(_QWORD **)(a1 + 40);
    if ( !*v31 || (v18 = 0, !((unsigned int (__fastcall *)(_QWORD, _QWORD))*v31)(v31[1], 0)) )
    {
      sub_2240CE0((__int64 *)&v43, (__int64)&v44[-1].m128i_i64[1] + 7, 1);
      if ( &_pthread_key_create )
      {
        v35 = pthread_mutex_lock(*(pthread_mutex_t **)(a1 + 64));
        if ( v35 )
          sub_4264C5(v35);
      }
      sub_2241490(*(unsigned __int64 **)(a1 + 72), v43, (size_t)v44);
      v36 = *(_QWORD *)(a1 + 80);
      v42[0] = v44;
      v20 = *(_BYTE **)(v36 + 8);
      if ( v20 == *(_BYTE **)(v36 + 16) )
      {
        sub_A235E0(v36, v20, v42);
      }
      else
      {
        if ( v20 )
        {
          *(_QWORD *)v20 = v44;
          v20 = *(_BYTE **)(v36 + 8);
        }
        v20 += 8;
        *(_QWORD *)(v36 + 8) = v20;
      }
      if ( &_pthread_key_create )
        pthread_mutex_unlock(*(pthread_mutex_t **)(a1 + 64));
      if ( v43 != (char *)v45 )
      {
        v20 = (_BYTE *)(v45[0] + 1LL);
        j_j___libc_free_0((unsigned __int64)v43);
      }
      goto LABEL_21;
    }
    if ( v43 != (char *)v45 )
    {
      v18 = v45[0] + 1LL;
      j_j___libc_free_0((unsigned __int64)v43);
    }
  }
  v32 = v41;
  if ( v41 )
  {
    sub_BA9C10(v41, v18, v19, v17);
    j_j___libc_free_0((unsigned __int64)v32);
  }
  v33 = v60;
  v28 = (unsigned __int64)&v60[48 * (unsigned int)v61];
  if ( v60 != (_BYTE *)v28 )
  {
    do
    {
      v28 -= 48LL;
      v34 = *(_QWORD *)(v28 + 16);
      if ( v34 != v28 + 32 )
        j_j___libc_free_0(v34);
    }
    while ( v33 != (_BYTE *)v28 );
    goto LABEL_29;
  }
LABEL_30:
  if ( (_BYTE *)v28 != v62 )
    _libc_free(v28);
  if ( v57 )
    j_j___libc_free_0(v57);
  if ( v54 != (_QWORD *)v56 )
    j_j___libc_free_0((unsigned __int64)v54);
  if ( v51 != (_QWORD *)v53 )
    j_j___libc_free_0((unsigned __int64)v51);
  if ( v47 != v49 )
    j_j___libc_free_0((unsigned __int64)v47);
  if ( v40 )
    (*(void (__fastcall **)(_QWORD *))(*v40 + 8LL))(v40);
}
