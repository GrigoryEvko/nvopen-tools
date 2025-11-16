// Function: sub_131A4E0
// Address: 0x131a4e0
//
__int64 __fastcall sub_131A4E0(struct __pthread_internal_list *a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rcx
  void *v4; // rsp
  unsigned __int64 i; // rax
  unsigned __int64 v6; // rbx
  unsigned int v7; // eax
  struct __pthread_internal_list *v8; // rax
  signed __int64 v9; // rsi
  __int64 v10; // rax
  pthread_mutex_t *v11; // rdx
  pthread_mutex_t *v12; // r14
  pthread_mutex_t *v13; // r12
  _QWORD *v14; // r13
  unsigned int v15; // ebx
  __int64 *v18; // [rsp+0h] [rbp-50h]
  struct __pthread_internal_list *v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  unsigned int v21; // [rsp+18h] [rbp-38h]
  int v22; // [rsp+1Ch] [rbp-34h]

  v2 = 0;
  v19 = a1;
  v3 = qword_5260D48[0];
  v4 = alloca(qword_5260D48[0]);
  for ( i = 0; i < v3; v2 = i )
  {
    *((_BYTE *)&v18 + i) = 0;
    i = (unsigned int)(v2 + 1);
  }
  LOBYTE(v18) = 1;
  v6 = 1;
  v7 = sub_1300B70(a1, a2, v2);
  v22 = 0;
  v21 = v7;
  v20 = v7;
  if ( v7 > 1 )
  {
    do
    {
      if ( !*((_BYTE *)&v18 + v6 % qword_5260D48[0]) && qword_50579C0[v6] )
      {
        v11 = (pthread_mutex_t *)(unk_5260DD8 + 208 * (v6 % qword_5260D48[0]));
        v12 = v11 + 3;
        v13 = v11;
        v18 = &v11[1].__align + 2;
        if ( pthread_mutex_trylock(v11 + 3) )
        {
          sub_130AD90((__int64)v18);
          v13[4].__size[0] = 1;
        }
        ++v13[2].__list.__next;
        v8 = v19;
        if ( v19 != v13[2].__list.__prev )
        {
          ++*(&v13[2].__align + 2);
          v13[2].__list.__prev = v8;
        }
        v13[4].__owner = 1;
        v13[4].__size[12] = 0;
        sub_130B0C0(&v13[4].__align + 2, 0);
        v13[4].__list.__prev = 0;
        v13[4].__list.__next = 0;
        sub_130B140(&v13[5].__align, &qword_4287700);
        ++unk_5260D40;
        v13[4].__size[0] = 0;
        pthread_mutex_unlock(v12);
        v9 = qword_5260D48[0];
        v10 = (unsigned int)++v22;
        *((_BYTE *)&v18 + v6 % qword_5260D48[0]) = 1;
        if ( v9 == v10 )
          break;
      }
      ++v6;
    }
    while ( v20 != v6 );
    if ( !(unsigned __int8)sub_131A1E0(v19, 0) )
      goto LABEL_14;
  }
  else if ( !(unsigned __int8)sub_131A1E0(v19, 0) )
  {
    if ( !v21 )
      return 0;
LABEL_14:
    v14 = qword_50579C0;
    v15 = 0;
    do
    {
      if ( *v14 )
        sub_130B8C0((__int64)v19, *v14 + 10648LL, 1u);
      ++v15;
      ++v14;
    }
    while ( v21 > v15 );
    return 0;
  }
  return 1;
}
