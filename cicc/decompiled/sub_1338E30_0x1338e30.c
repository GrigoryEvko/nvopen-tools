// Function: sub_1338E30
// Address: 0x1338e30
//
int __fastcall sub_1338E30(__int64 a1, int a2, unsigned __int8 a3)
{
  unsigned int v5; // edx
  void *v6; // rsp
  __int64 *v7; // rcx
  __int64 v8; // r15
  __int64 *v9; // rbx
  __int64 *v10; // rdx
  __int64 v11; // r15
  __int64 *i; // rax
  int result; // eax
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v17; // [rsp+8h] [rbp-38h]

  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_QWORD)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  v5 = *(_DWORD *)(qword_4F96BA0 + 8);
  if ( a2 == 4096 || v5 == a2 )
  {
    v6 = alloca(16 * ((8 * (unsigned __int64)v5 + 15) >> 4));
    if ( v5 )
    {
      v7 = qword_50579C0;
      v8 = v5 - 1;
      v9 = &v16;
      v10 = &v16;
      v11 = v8;
      for ( i = &qword_50579C0[1]; ; ++i )
      {
        *v10++ = *v7;
        v7 = i;
        if ( i == &qword_50579C0[v11 + 1] )
          break;
      }
      v17 = &v16;
      byte_4F96C28 = 0;
      result = pthread_mutex_unlock(&stru_4F96C00);
      v14 = (__int64)&v17[v11 + 1];
      do
      {
        if ( *v9 )
          result = sub_1315160(a1, *v9, 0, a3);
        ++v9;
      }
      while ( (__int64 *)v14 != v9 );
    }
    else
    {
      byte_4F96C28 = 0;
      return pthread_mutex_unlock(&stru_4F96C00);
    }
  }
  else
  {
    v15 = qword_50579C0[a2];
    byte_4F96C28 = 0;
    result = pthread_mutex_unlock(&stru_4F96C00);
    if ( v15 )
      return sub_1315160(a1, v15, 0, a3);
  }
  return result;
}
