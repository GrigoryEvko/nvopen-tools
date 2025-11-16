// Function: sub_23CCA50
// Address: 0x23cca50
//
int __fastcall sub_23CCA50(__int64 a1, int a2)
{
  pthread_t *v4; // r15
  unsigned __int64 v5; // r9
  unsigned __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int32 v9; // esi
  __int64 v10; // rbx
  bool v11; // cf
  unsigned __int64 v12; // rbx
  _QWORD *v13; // r8
  pthread_t *v14; // rdx
  pthread_t *v15; // rax
  pthread_t v16; // rcx
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  unsigned __int64 v20; // [rsp+8h] [rbp-58h]
  unsigned __int64 v21; // [rsp+8h] [rbp-58h]
  _QWORD *v22; // [rsp+10h] [rbp-50h]
  _QWORD *v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  pthread_rwlock_t *rwlock; // [rsp+18h] [rbp-48h]
  __m128i v26[4]; // [rsp+20h] [rbp-40h] BYREF

  rwlock = (pthread_rwlock_t *)(a1 + 32);
  if ( &_pthread_key_create && pthread_rwlock_wrlock((pthread_rwlock_t *)(a1 + 32)) == 35 )
    sub_4264C5(0x23u);
  v4 = *(pthread_t **)(a1 + 16);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(unsigned int *)(a1 + 364);
  v7 = *(_QWORD *)(a1 + 16) - v5;
  v8 = v7 >> 3;
  if ( v6 > v7 >> 3 )
  {
    v9 = v7 >> 3;
    if ( (int)v6 <= a2 )
      a2 = v6;
    if ( a2 > (int)v8 )
    {
      while ( 1 )
      {
        v26[0].m128i_i64[0] = a1;
        v26[0].m128i_i32[2] = v9;
        if ( *(pthread_t **)(a1 + 24) != v4 )
        {
          if ( v4 )
            sub_23CC7F0(v4, unk_3F67788, v26);
          v5 = *(_QWORD *)(a1 + 8);
          v4 = (pthread_t *)(*(_QWORD *)(a1 + 16) + 8LL);
          *(_QWORD *)(a1 + 16) = v4;
          goto LABEL_11;
        }
        if ( v8 == 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v10 = 1;
        if ( v8 )
          v10 = v8;
        v11 = __CFADD__(v8, v10);
        v12 = v8 + v10;
        if ( v11 )
          break;
        if ( v12 )
        {
          if ( v12 > 0xFFFFFFFFFFFFFFFLL )
            v12 = 0xFFFFFFFFFFFFFFFLL;
          v17 = 8 * v12;
          goto LABEL_33;
        }
        v13 = 0;
LABEL_19:
        if ( (_QWORD *)((char *)v13 + v7) )
        {
          v20 = v5;
          v22 = v13;
          sub_23CC7F0((_QWORD *)((char *)v13 + v7), unk_3F67788, v26);
          v5 = v20;
          v13 = v22;
        }
        v14 = v13;
        if ( v4 != (pthread_t *)v5 )
        {
          v15 = (pthread_t *)v5;
          do
          {
            if ( v14 )
            {
              v16 = *v15;
              *v15 = 0;
              *v14 = v16;
            }
            if ( *v15 )
              sub_2207530();
            ++v15;
            ++v14;
          }
          while ( v4 != v15 );
        }
        v4 = v14 + 1;
        if ( v5 )
        {
          v23 = v13;
          j_j___libc_free_0(v5);
          v13 = v23;
        }
        *(_QWORD *)(a1 + 8) = v13;
        v5 = (unsigned __int64)v13;
        *(_QWORD *)(a1 + 16) = v4;
        *(_QWORD *)(a1 + 24) = &v13[v12];
LABEL_11:
        v7 = (__int64)v4 - v5;
        v8 = (__int64)((__int64)v4 - v5) >> 3;
        v9 = v8;
        if ( (int)v8 >= a2 )
          goto LABEL_30;
      }
      v17 = 0x7FFFFFFFFFFFFFF8LL;
      v12 = 0xFFFFFFFFFFFFFFFLL;
LABEL_33:
      v21 = v5;
      v24 = v7;
      v18 = sub_22077B0(v17);
      v7 = v24;
      v5 = v21;
      v13 = (_QWORD *)v18;
      goto LABEL_19;
    }
  }
LABEL_30:
  if ( &_pthread_key_create )
    LODWORD(v8) = pthread_rwlock_unlock(rwlock);
  return v8;
}
