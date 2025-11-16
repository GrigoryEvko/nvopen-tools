// Function: sub_131A910
// Address: 0x131a910
//
int __fastcall sub_131A910(__int64 a1)
{
  unsigned int v1; // r14d
  __int64 v2; // rax
  int result; // eax
  __int64 v4; // rax
  pthread_mutex_t *v5; // rdx
  pthread_mutex_t *v6; // r15
  pthread_mutex_t *v7; // r14
  __int64 v8; // [rsp+0h] [rbp-40h]
  unsigned int v9; // [rsp+Ch] [rbp-34h]

  if ( qword_5260D48[0] )
  {
    v1 = 0;
    v2 = 0;
    do
    {
      sub_130B060(a1, (__int64 *)(qword_5260DD8 + 208 * v2 + 56));
      v2 = ++v1;
    }
    while ( (unsigned __int64)v1 < qword_5260D48[0] );
  }
  result = (unsigned int)sub_130B060(a1, qword_5260D60);
  if ( byte_4F96B80 )
  {
    if ( pthread_mutex_trylock(&stru_5260DA0) )
    {
      sub_130AD90((__int64)qword_5260D60);
      unk_5260DC8 = 1;
    }
    ++unk_5260D98;
    if ( a1 != unk_5260D90 )
    {
      ++unk_5260D88;
      unk_5260D90 = a1;
    }
    unk_5260D40 = 0;
    byte_5260DD0[0] = 0;
    if ( qword_5260D48[0] )
    {
      v9 = 0;
      v4 = 0;
      do
      {
        v5 = (pthread_mutex_t *)(qword_5260DD8 + 208 * v4);
        v6 = v5 + 3;
        v7 = v5;
        v8 = (__int64)(&v5[1].__align + 2);
        if ( pthread_mutex_trylock(v5 + 3) )
        {
          sub_130AD90(v8);
          v7[4].__size[0] = 1;
        }
        ++v7[2].__list.__next;
        if ( (struct __pthread_internal_list *)a1 != v7[2].__list.__prev )
        {
          ++*(&v7[2].__align + 2);
          v7[2].__list.__prev = (struct __pthread_internal_list *)a1;
        }
        v7[4].__owner = 0;
        pthread_cond_init((pthread_cond_t *)(&v7->__align + 1), 0);
        v7[4].__size[12] = 0;
        sub_130B0C0(&v7[4].__align + 2, 0);
        v7[4].__list.__prev = 0;
        v7[4].__list.__next = 0;
        sub_130B140(&v7[5].__align, &qword_4287700);
        v7[4].__size[0] = 0;
        pthread_mutex_unlock(v6);
        v4 = ++v9;
      }
      while ( (unsigned __int64)v9 < qword_5260D48[0] );
    }
    unk_5260DC8 = 0;
    return pthread_mutex_unlock(&stru_5260DA0);
  }
  return result;
}
