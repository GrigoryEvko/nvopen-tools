// Function: sub_C99310
// Address: 0xc99310
//
int __fastcall sub_C99310(__int64 a1, __int64 a2)
{
  __int64 **v2; // rax
  __int64 *v3; // r12
  _QWORD *v4; // rax
  __int64 v5; // rsi
  __int128 *v6; // rax
  unsigned int v7; // eax
  __int64 align; // rax
  __int64 *v9; // r14
  __int64 *v10; // rdi
  __int64 *v11; // rdi
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rbx
  _QWORD *v16; // rdi
  __int64 *v17; // rbx
  __int64 *v18; // r12
  __int64 *v19; // rdi
  __int64 *v20; // rdi
  __int64 *v21; // rdi
  __int64 v22; // r13
  _QWORD *v23; // r12
  _QWORD *v24; // rbx
  _QWORD *v25; // r15
  _QWORD *v26; // rdi
  _QWORD *v27; // rdi
  _QWORD *v28; // rdi
  _QWORD *v29; // rdi
  _QWORD *v30; // rdi
  _QWORD *v31; // rdi
  pthread_mutex_t *mutex; // [rsp+0h] [rbp-50h]
  __int64 **v34; // [rsp+8h] [rbp-48h]
  __int64 **v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v2 = (__int64 **)sub_C94E20((__int64)&qword_4F84F00);
  v3 = (__int64 *)qword_4F84F10;
  if ( v2 )
    v3 = *v2;
  if ( v3 )
  {
    sub_C99050(v3, a2);
    j_j___libc_free_0(v3, 16672);
  }
  v4 = (_QWORD *)sub_CEECD0(8, 8);
  *v4 = 0;
  v5 = (__int64)v4;
  sub_C94E10((__int64)&qword_4F84F00, v4);
  v6 = sub_C95C80();
  mutex = (pthread_mutex_t *)v6;
  if ( &_pthread_key_create )
  {
    v7 = pthread_mutex_lock((pthread_mutex_t *)v6);
    if ( v7 )
      sub_4264C5(v7);
  }
  align = mutex[1].__align;
  v34 = (__int64 **)*(&mutex[1].__align + 1);
  if ( v34 != (__int64 **)align )
  {
    v35 = (__int64 **)mutex[1].__align;
    do
    {
      v9 = *v35;
      if ( *v35 )
      {
        v10 = (__int64 *)v9[2078];
        if ( v10 != v9 + 2081 )
          _libc_free(v10, v5);
        v11 = (__int64 *)v9[2073];
        if ( v11 != v9 + 2075 )
        {
          v5 = v9[2075] + 1;
          j_j___libc_free_0(v11, v5);
        }
        v12 = v9[2068];
        if ( *((_DWORD *)v9 + 4139) )
        {
          v13 = *((unsigned int *)v9 + 4138);
          if ( (_DWORD)v13 )
          {
            v14 = 8 * v13;
            v15 = 0;
            do
            {
              v16 = *(_QWORD **)(v12 + v15);
              if ( v16 != (_QWORD *)-8LL && v16 )
              {
                v5 = *v16 + 25LL;
                sub_C7D6A0((__int64)v16, v5, 8);
                v12 = v9[2068];
              }
              v15 += 8;
            }
            while ( v15 != v14 );
          }
        }
        _libc_free(v12, v5);
        v17 = (__int64 *)v9[18];
        v18 = &v17[16 * (unsigned __int64)*((unsigned int *)v9 + 38)];
        if ( v17 != v18 )
        {
          do
          {
            v18 -= 16;
            v19 = (__int64 *)v18[10];
            if ( v19 != v18 + 12 )
            {
              v5 = v18[12] + 1;
              j_j___libc_free_0(v19, v5);
            }
            v20 = (__int64 *)v18[6];
            if ( v20 != v18 + 8 )
            {
              v5 = v18[8] + 1;
              j_j___libc_free_0(v20, v5);
            }
            v21 = (__int64 *)v18[2];
            if ( v21 != v18 + 4 )
            {
              v5 = v18[4] + 1;
              j_j___libc_free_0(v21, v5);
            }
          }
          while ( v17 != v18 );
          v18 = (__int64 *)v9[18];
        }
        if ( v18 != v9 + 20 )
          _libc_free(v18, v5);
        v22 = *v9 + 8LL * *((unsigned int *)v9 + 2);
        v36 = *v9;
        if ( *v9 != v22 )
        {
          do
          {
            v23 = *(_QWORD **)(v22 - 8);
            v22 -= 8;
            if ( v23 )
            {
              v24 = (_QWORD *)v23[17];
              v25 = (_QWORD *)v23[16];
              if ( v24 != v25 )
              {
                do
                {
                  v26 = (_QWORD *)v25[10];
                  if ( v26 != v25 + 12 )
                    j_j___libc_free_0(v26, v25[12] + 1LL);
                  v27 = (_QWORD *)v25[6];
                  if ( v27 != v25 + 8 )
                    j_j___libc_free_0(v27, v25[8] + 1LL);
                  v28 = (_QWORD *)v25[2];
                  if ( v28 != v25 + 4 )
                    j_j___libc_free_0(v28, v25[4] + 1LL);
                  v25 += 16;
                }
                while ( v24 != v25 );
                v25 = (_QWORD *)v23[16];
              }
              if ( v25 )
                j_j___libc_free_0(v25, v23[18] - (_QWORD)v25);
              v29 = (_QWORD *)v23[10];
              if ( v29 != v23 + 12 )
                j_j___libc_free_0(v29, v23[12] + 1LL);
              v30 = (_QWORD *)v23[6];
              if ( v30 != v23 + 8 )
                j_j___libc_free_0(v30, v23[8] + 1LL);
              v31 = (_QWORD *)v23[2];
              if ( v31 != v23 + 4 )
                j_j___libc_free_0(v31, v23[4] + 1LL);
              v5 = 152;
              j_j___libc_free_0(v23, 152);
            }
          }
          while ( v36 != v22 );
          v22 = *v9;
        }
        if ( (__int64 *)v22 != v9 + 2 )
          _libc_free(v22, v5);
        v5 = 16672;
        j_j___libc_free_0(v9, 16672);
      }
      ++v35;
    }
    while ( v34 != v35 );
    align = mutex[1].__align;
    if ( align != *(&mutex[1].__align + 1) )
      *(&mutex[1].__align + 1) = align;
  }
  if ( &_pthread_key_create )
    LODWORD(align) = pthread_mutex_unlock(mutex);
  return align;
}
