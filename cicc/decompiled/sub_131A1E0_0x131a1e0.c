// Function: sub_131A1E0
// Address: 0x131a1e0
//
__int64 __fastcall sub_131A1E0(struct __pthread_internal_list *a1, unsigned int a2)
{
  unsigned __int64 v2; // rbx
  pthread_mutex_t *v3; // rcx
  pthread_mutex_t *v4; // r12
  pthread_mutex_t *v5; // r15
  unsigned int v6; // r14d
  int v8; // eax
  bool v9; // zf
  int v10; // ebx
  __int64 v11; // rbx
  pthread_mutex_t *v12; // r12
  __int64 v13; // r14
  __int64 v14; // [rsp+0h] [rbp-40h]

  v2 = (unsigned __int64)a2 % qword_5260D48[0];
  v3 = (pthread_mutex_t *)(unk_5260DD8 + 208 * v2);
  v4 = v3 + 3;
  v5 = v3;
  v14 = (__int64)(&v3[1].__align + 2);
  if ( pthread_mutex_trylock(v3 + 3) )
  {
    sub_130AD90(v14);
    v5[4].__size[0] = 1;
  }
  ++v5[2].__list.__next;
  if ( a1 != v5[2].__list.__prev )
  {
    ++*(&v5[2].__align + 2);
    v5[2].__list.__prev = a1;
  }
  v6 = byte_5260DD0[0];
  if ( !byte_5260DD0[0] || v5[4].__owner )
  {
    v5[4].__size[0] = 0;
    pthread_mutex_unlock(v4);
    return 0;
  }
  v5[4].__owner = 1;
  v5[4].__size[12] = 0;
  sub_130B0C0(&v5[4].__align + 2, 0);
  v5[4].__list.__prev = 0;
  v5[4].__list.__next = 0;
  sub_130B140(&v5[5].__align, &qword_4287700);
  ++unk_5260D40;
  v5[4].__size[0] = 0;
  pthread_mutex_unlock(v4);
  if ( a2 )
  {
    v11 = unk_5260DD8;
    v12 = (pthread_mutex_t *)(unk_5260DD8 + 120LL);
    v13 = unk_5260DD8 + 56LL;
    if ( pthread_mutex_trylock((pthread_mutex_t *)(unk_5260DD8 + 120LL)) )
    {
      sub_130AD90(v13);
      *(_BYTE *)(v11 + 160) = 1;
    }
    ++*(_QWORD *)(v11 + 112);
    if ( a1 != *(struct __pthread_internal_list **)(v11 + 104) )
    {
      ++*(_QWORD *)(v11 + 96);
      *(_QWORD *)(v11 + 104) = a1;
    }
    v6 = 0;
    pthread_cond_signal((pthread_cond_t *)(v11 + 8));
    *(_BYTE *)(v11 + 160) = 0;
    pthread_mutex_unlock(v12);
  }
  else
  {
    ++BYTE1(a1->__prev);
    if ( !LOBYTE(a1[51].__prev) )
      sub_1313A40(a1);
    v8 = sub_1319830(v5, v2);
    v9 = BYTE1(a1->__prev)-- == 1;
    v10 = v8;
    if ( v9 )
      sub_1313A40(a1);
    if ( !v10 )
      return 0;
    sub_130ACF0("<jemalloc>: arena 0 background thread creation failed (%d)\n", v10);
    if ( pthread_mutex_trylock(v4) )
    {
      sub_130AD90(v14);
      v5[4].__size[0] = 1;
    }
    ++v5[2].__list.__next;
    if ( a1 != v5[2].__list.__prev )
    {
      ++*(&v5[2].__align + 2);
      v5[2].__list.__prev = a1;
    }
    v5[4].__owner = 0;
    --unk_5260D40;
    v5[4].__size[0] = 0;
    pthread_mutex_unlock(v4);
  }
  return v6;
}
