// Function: sub_133B020
// Address: 0x133b020
//
__int64 __fastcall sub_133B020(__int64 a1)
{
  __int64 v1; // rdx
  int v2; // edx
  pthread_mutex_t **v3; // rax
  _QWORD *v4; // rsi
  pthread_mutex_t *v5; // rax
  pthread_mutex_t *v6; // r12
  _OWORD *v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rax
  char *v10; // r14
  unsigned int v11; // ebx
  __int64 v12; // r15
  _QWORD *v14; // [rsp+8h] [rbp-58h]
  _QWORD *v15; // [rsp+10h] [rbp-50h]
  pthread_mutex_t *v16; // [rsp+20h] [rbp-40h]
  unsigned int *v17; // [rsp+28h] [rbp-38h]

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
  sub_130AEF0(a1, &xmmword_4F96BC0);
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
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
  sub_130AEF0(a1, qword_5260D60);
  unk_5260DC8 = 0;
  pthread_mutex_unlock(&stru_5260DA0);
  v2 = sub_1300B70(&stru_5260DA0, qword_5260D60, v1);
  if ( v2 )
  {
    v3 = (pthread_mutex_t **)qword_50579C0;
    v15 = &qword_50579C0[1];
    v14 = &qword_50579C0[(unsigned int)(v2 - 1) + 1];
    while ( 1 )
    {
      v5 = *v3;
      v16 = v5;
      if ( v5 )
      {
        v6 = v5 + 265;
        v7 = &v5[263].__align + 2;
        if ( pthread_mutex_trylock(v5 + 265) )
        {
          sub_130AD90((__int64)v7);
          v16[266].__size[0] = 1;
        }
        ++v16[264].__list.__next;
        if ( (struct __pthread_internal_list *)a1 != v16[264].__list.__prev )
        {
          ++*(&v16[264].__align + 2);
          v16[264].__list.__prev = (struct __pthread_internal_list *)a1;
        }
        sub_130AEF0(a1, v7);
        v16[266].__size[0] = 0;
        pthread_mutex_unlock(v6);
        if ( pthread_mutex_trylock((pthread_mutex_t *)((char *)v16 + 78832)) )
        {
          sub_130AD90((__int64)(&v16[1969].__align + 1));
          v16[1971].__size[32] = 1;
        }
        ++v16[1970].__list.__prev;
        if ( a1 != *(&v16[1970].__align + 2) )
        {
          ++*(&v16[1970].__align + 1);
          *(&v16[1970].__align + 2) = a1;
        }
        sub_130AEF0(a1, &v16[1969].__align + 1);
        v16[1971].__size[32] = 0;
        pthread_mutex_unlock((pthread_mutex_t *)((char *)v16 + 78832));
        if ( pthread_mutex_trylock((pthread_mutex_t *)((char *)v16 + 10792)) )
        {
          sub_130AD90((__int64)(&v16[268].__align + 1));
          v16[270].__size[32] = 1;
        }
        ++v16[269].__list.__prev;
        if ( a1 != *(&v16[269].__align + 2) )
        {
          ++*(&v16[269].__align + 1);
          *(&v16[269].__align + 2) = a1;
        }
        sub_130AEF0(a1, &v16[268].__align + 1);
        v16[270].__size[32] = 0;
        pthread_mutex_unlock((pthread_mutex_t *)((char *)v16 + 10792));
        if ( pthread_mutex_trylock((pthread_mutex_t *)((char *)v16 + 30232)) )
        {
          sub_130AD90((__int64)(&v16[754].__align + 1));
          v16[756].__size[32] = 1;
        }
        ++v16[755].__list.__prev;
        if ( a1 != *(&v16[755].__align + 2) )
        {
          ++*(&v16[755].__align + 1);
          *(&v16[755].__align + 2) = a1;
        }
        sub_130AEF0(a1, &v16[754].__align + 1);
        v16[756].__size[32] = 0;
        pthread_mutex_unlock((pthread_mutex_t *)((char *)v16 + 30232));
        if ( pthread_mutex_trylock((pthread_mutex_t *)((char *)v16 + 49672)) )
        {
          sub_130AD90((__int64)(&v16[1240].__align + 1));
          v16[1242].__size[32] = 1;
        }
        ++v16[1241].__list.__prev;
        if ( a1 != *(&v16[1241].__align + 2) )
        {
          ++*(&v16[1241].__align + 1);
          *(&v16[1241].__align + 2) = a1;
        }
        sub_130AEF0(a1, &v16[1240].__align + 1);
        v16[1242].__size[32] = 0;
        pthread_mutex_unlock((pthread_mutex_t *)((char *)v16 + 49672));
        if ( pthread_mutex_trylock((pthread_mutex_t *)((char *)v16 + 69384)) )
        {
          sub_130AD90((__int64)&v16[1733]);
          v16[1735].__size[24] = 1;
        }
        ++*(&v16[1734].__align + 2);
        if ( a1 != *(&v16[1734].__align + 1) )
        {
          ++v16[1734].__align;
          *(&v16[1734].__align + 1) = a1;
        }
        sub_130AEF0(a1, &v16[1733].__lock);
        v16[1735].__size[24] = 0;
        pthread_mutex_unlock((pthread_mutex_t *)((char *)v16 + 69384));
        if ( pthread_mutex_trylock((pthread_mutex_t *)((char *)v16 + 71168)) )
        {
          sub_130AD90((__int64)&v16[1777].__list);
          v16[1780].__size[8] = 1;
        }
        ++v16[1779].__align;
        if ( (struct __pthread_internal_list *)a1 != v16[1778].__list.__next )
        {
          ++v16[1778].__list.__prev;
          v16[1778].__list.__next = (struct __pthread_internal_list *)a1;
        }
        sub_130AEF0(a1, &v16[1777].__align + 3);
        v16[1780].__size[8] = 0;
        pthread_mutex_unlock((pthread_mutex_t *)((char *)v16 + 71168));
        if ( pthread_mutex_trylock((pthread_mutex_t *)((char *)v16 + 10472)) )
        {
          sub_130AD90((__int64)(&v16[260].__align + 1));
          v16[262].__size[32] = 1;
        }
        ++v16[261].__list.__prev;
        if ( a1 != *(&v16[261].__align + 2) )
        {
          ++*(&v16[261].__align + 1);
          *(&v16[261].__align + 2) = a1;
        }
        sub_130AEF0(a1, &v16[260].__align + 1);
        v16[262].__size[32] = 0;
        pthread_mutex_unlock((pthread_mutex_t *)((char *)v16 + 10472));
        v8 = *(&v16[1973].__align + 2);
        if ( pthread_mutex_trylock((pthread_mutex_t *)(v8 + 96)) )
        {
          sub_130AD90(v8 + 32);
          *(_BYTE *)(v8 + 136) = 1;
        }
        ++*(_QWORD *)(v8 + 88);
        if ( a1 != *(_QWORD *)(v8 + 80) )
        {
          ++*(_QWORD *)(v8 + 72);
          *(_QWORD *)(v8 + 80) = a1;
        }
        sub_130AEF0(a1, (_OWORD *)(*(&v16[1973].__align + 2) + 32));
        v9 = *(&v16[1973].__align + 2);
        *(_BYTE *)(v9 + 136) = 0;
        pthread_mutex_unlock((pthread_mutex_t *)(v9 + 96));
        v10 = (char *)&unk_5260DF4;
        v17 = dword_5060A40;
        do
        {
          v11 = 0;
          while ( *(_DWORD *)v10 > v11 )
          {
            v12 = (__int64)v16 + 224 * v11 + *v17;
            if ( pthread_mutex_trylock((pthread_mutex_t *)(v12 + 64)) )
            {
              sub_130AD90(v12);
              *(_BYTE *)(v12 + 104) = 1;
            }
            ++*(_QWORD *)(v12 + 56);
            if ( a1 != *(_QWORD *)(v12 + 48) )
            {
              ++*(_QWORD *)(v12 + 40);
              *(_QWORD *)(v12 + 48) = a1;
            }
            ++v11;
            sub_130AEF0(a1, (_OWORD *)v12);
            *(_BYTE *)(v12 + 104) = 0;
            pthread_mutex_unlock((pthread_mutex_t *)(v12 + 64));
          }
          ++v17;
          v10 += 40;
        }
        while ( (char *)&unk_5260DE0 + 1460 != v10 );
        v4 = v15;
        v3 = (pthread_mutex_t **)v15;
        if ( v14 == v15 )
          return 0;
      }
      else
      {
        v4 = v15;
        v3 = (pthread_mutex_t **)v15;
        if ( v14 == v15 )
          return 0;
      }
      v15 = v4 + 1;
    }
  }
  return 0;
}
