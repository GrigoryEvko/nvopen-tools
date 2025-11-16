// Function: sub_16DB140
// Address: 0x16db140
//
int sub_16DB140()
{
  __int64 *v0; // rax
  __int64 v1; // r12
  unsigned int v2; // eax
  __int64 v3; // r14
  __int64 *v4; // r13
  __int64 v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v7; // rdi
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rbx
  _QWORD *v14; // r15
  _QWORD *v15; // rdi
  _QWORD *v16; // rdi
  _QWORD *v17; // rbx
  _QWORD *v18; // r15
  _QWORD *v19; // rdi
  _QWORD *v20; // rdi
  _QWORD *v21; // rax
  int result; // eax

  v0 = (__int64 *)sub_16D40F0((__int64)&qword_4FA1650);
  v1 = qword_4FA1660;
  if ( v0 )
    v1 = *v0;
  if ( v1 )
  {
    sub_16DAF90(v1);
    j_j___libc_free_0(v1, 11672);
  }
  if ( &_pthread_key_create )
  {
    v2 = pthread_mutex_lock(&stru_4FA16A0);
    if ( v2 )
      sub_4264C5(v2);
  }
  v3 = qword_4FA1678;
  v4 = (__int64 *)qword_4FA1670;
  if ( qword_4FA1678 != qword_4FA1670 )
  {
    do
    {
      v5 = *v4;
      if ( *v4 )
      {
        v6 = *(_QWORD *)(v5 + 11640);
        if ( v6 != v5 + 11656 )
          _libc_free(v6);
        v7 = *(_QWORD *)(v5 + 11600);
        if ( v7 != v5 + 11616 )
          j_j___libc_free_0(v7, *(_QWORD *)(v5 + 11616) + 1LL);
        v8 = *(_QWORD *)(v5 + 11552);
        if ( *(_DWORD *)(v5 + 11564) )
        {
          v9 = *(unsigned int *)(v5 + 11560);
          if ( (_DWORD)v9 )
          {
            v10 = 8 * v9;
            v11 = 0;
            do
            {
              v12 = *(_QWORD *)(v8 + v11);
              if ( v12 != -8 && v12 )
              {
                _libc_free(v12);
                v8 = *(_QWORD *)(v5 + 11552);
              }
              v11 += 8;
            }
            while ( v10 != v11 );
          }
        }
        _libc_free(v8);
        v13 = *(_QWORD **)(v5 + 1296);
        v14 = &v13[10 * *(unsigned int *)(v5 + 1304)];
        if ( v13 != v14 )
        {
          do
          {
            v14 -= 10;
            v15 = (_QWORD *)v14[6];
            if ( v15 != v14 + 8 )
              j_j___libc_free_0(v15, v14[8] + 1LL);
            v16 = (_QWORD *)v14[2];
            if ( v16 != v14 + 4 )
              j_j___libc_free_0(v16, v14[4] + 1LL);
          }
          while ( v13 != v14 );
          v14 = *(_QWORD **)(v5 + 1296);
        }
        if ( v14 != (_QWORD *)(v5 + 1312) )
          _libc_free((unsigned __int64)v14);
        v17 = *(_QWORD **)v5;
        v18 = (_QWORD *)(*(_QWORD *)v5 + 80LL * *(unsigned int *)(v5 + 8));
        if ( *(_QWORD **)v5 != v18 )
        {
          do
          {
            v18 -= 10;
            v19 = (_QWORD *)v18[6];
            if ( v19 != v18 + 8 )
              j_j___libc_free_0(v19, v18[8] + 1LL);
            v20 = (_QWORD *)v18[2];
            if ( v20 != v18 + 4 )
              j_j___libc_free_0(v20, v18[4] + 1LL);
          }
          while ( v17 != v18 );
          v18 = *(_QWORD **)v5;
        }
        if ( v18 != (_QWORD *)(v5 + 16) )
          _libc_free((unsigned __int64)v18);
        j_j___libc_free_0(v5, 11672);
      }
      ++v4;
    }
    while ( (__int64 *)v3 != v4 );
    if ( qword_4FA1670 != qword_4FA1678 )
      qword_4FA1678 = qword_4FA1670;
  }
  v21 = (_QWORD *)sub_1C42D70(8, 8);
  *v21 = 0;
  result = sub_16D40E0((__int64)&qword_4FA1650, v21);
  if ( &_pthread_key_create )
    return pthread_mutex_unlock(&stru_4FA16A0);
  return result;
}
