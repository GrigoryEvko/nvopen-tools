// Function: sub_13198E0
// Address: 0x13198e0
//
__int64 __fastcall sub_13198E0(signed int a1)
{
  _BYTE *v2; // r12
  pthread_mutex_t *v3; // rax
  __int64 v4; // rax
  __int64 v5; // r14
  _QWORD *v6; // rdi
  int owner; // eax
  __int64 v8; // r15
  unsigned int v9; // r12d
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  void *v17; // rsp
  __int64 v18; // r15
  unsigned int v19; // edx
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // r14
  unsigned int v23; // ebx
  __int64 v24; // r15
  unsigned int v25; // r12d
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rsi
  unsigned int v30; // ebx
  __int64 v31; // rdx
  __int64 v32; // rax
  signed __int64 *v33; // r14
  __int64 v34; // r13
  unsigned int v35; // r14d
  __int64 v36; // rbx
  char *v37; // r12
  __int64 v38; // r15
  int v39; // eax
  pthread_mutex_t *v40; // r8
  _BYTE *v41; // rax
  int v42; // r13d
  char v43; // bl
  __int64 v44; // r15
  __int64 v45; // r14
  __int64 v46; // r13
  char *v47; // rbx
  int v48; // eax
  bool v49; // zf
  int v50; // r14d
  _BYTE v51[8]; // [rsp+0h] [rbp-110h] BYREF
  _BYTE *v52; // [rsp+8h] [rbp-108h]
  _QWORD *v53; // [rsp+10h] [rbp-100h]
  __int64 *v54; // [rsp+18h] [rbp-F8h]
  pthread_mutex_t *v55; // [rsp+20h] [rbp-F0h]
  pthread_mutex_t *mutex; // [rsp+28h] [rbp-E8h]
  char *v57; // [rsp+30h] [rbp-E0h]
  unsigned int v58; // [rsp+3Ch] [rbp-D4h]
  pthread_mutex_t *v59; // [rsp+40h] [rbp-D0h]
  _BYTE *v60; // [rsp+48h] [rbp-C8h]
  pthread_mutex_t *v61; // [rsp+50h] [rbp-C0h]
  signed __int64 *v62; // [rsp+58h] [rbp-B8h]
  cpu_set_t cpuset; // [rsp+60h] [rbp-B0h] BYREF

  v58 = a1;
  if ( unk_4C6F238 != 2 )
  {
    memset(&cpuset, 0, sizeof(cpuset));
    if ( (unsigned __int64)a1 <= 0x3FF )
      cpuset.__bits[(unsigned __int64)a1 >> 6] |= 1LL << a1;
    sched_setaffinity(0, 0x80u, &cpuset);
  }
  v2 = (_BYTE *)(__readfsqword(0) - 2664);
  if ( __readfsbyte(0xFFFFF8C8) )
    v2 = (_BYTE *)sub_1313D30((__int64)v2, 1);
  sub_1313AB0((__int64)v2, 5u);
  v3 = (pthread_mutex_t *)(unk_5260DD8 + 208LL * (unsigned int)a1);
  v59 = v3;
  v54 = &v3[1].__align + 2;
  mutex = v3 + 3;
  if ( pthread_mutex_trylock(v3 + 3) )
  {
    sub_130AD90((__int64)v54);
    v59[4].__size[0] = 1;
  }
  v4 = (__int64)v59;
  ++v59[2].__list.__next;
  if ( v2 != *(_BYTE **)(v4 + 104) )
  {
    ++*(_QWORD *)(v4 + 96);
    *(_QWORD *)(v4 + 104) = v2;
  }
  v5 = (__int64)v59;
  v57 = &v59[4].__size[12];
  v6 = &v59[4].__align + 2;
  v59[4].__size[12] = 1;
  v53 = v6;
  sub_130B0C0(v6, -1);
  if ( !a1 )
  {
    v52 = v51;
    v16 = qword_5260D48[0];
    v62 = qword_5260D48;
    v17 = alloca(qword_5260D48[0]);
    v57 = v51;
    v18 = unk_5260DD8;
    if ( qword_5260D48[0] <= 1uLL )
    {
      v21 = *(_DWORD *)(unk_5260DD8 + 168LL);
      if ( !v21 )
      {
LABEL_69:
        *(_DWORD *)(v18 + 168) = 0;
        v55 = v59 + 4;
        goto LABEL_24;
      }
    }
    else
    {
      v19 = 1;
      v20 = 1;
      do
      {
        v51[v20] = 0;
        v20 = ++v19;
      }
      while ( v16 > v19 );
      v21 = *(_DWORD *)(v18 + 168);
      if ( !v21 )
      {
LABEL_57:
        v30 = 1;
        v31 = 1;
        v32 = 1;
        v33 = v62;
        do
        {
          v34 = v18 + 208 * v31;
          if ( v57[v32] )
          {
            sub_13196F0(v2, v18 + 208 * v31);
          }
          else
          {
            v62 = (signed __int64 *)(v34 + 56);
            if ( pthread_mutex_trylock((pthread_mutex_t *)(v34 + 120)) )
            {
              sub_130AD90((__int64)v62);
              *(_BYTE *)(v34 + 160) = 1;
            }
            ++*(_QWORD *)(v34 + 112);
            if ( v2 != *(_BYTE **)(v34 + 104) )
            {
              ++*(_QWORD *)(v34 + 96);
              *(_QWORD *)(v34 + 104) = v2;
            }
            if ( *(_DWORD *)(v34 + 168) )
            {
              *(_DWORD *)(v34 + 168) = 0;
              --unk_5260D40;
            }
            *(_BYTE *)(v34 + 160) = 0;
            pthread_mutex_unlock((pthread_mutex_t *)(v34 + 120));
          }
          v32 = ++v30;
          v31 = v30;
          v18 = unk_5260DD8;
        }
        while ( v30 < (unsigned __int64)*v33 );
        goto LABEL_69;
      }
    }
    LODWORD(v55) = 1;
    do
    {
      if ( v21 == 2 )
      {
        *(_BYTE *)(v18 + 160) = 0;
        pthread_mutex_unlock((pthread_mutex_t *)(v18 + 120));
        if ( pthread_mutex_trylock(&stru_5260DA0) )
        {
          sub_130AD90((__int64)&unk_5260D60);
          unk_5260DC8 = 1;
        }
        ++unk_5260D98;
        if ( v2 != (_BYTE *)unk_5260D90 )
        {
          ++unk_5260D88;
          unk_5260D90 = v2;
        }
        unk_5260DC8 = 0;
        pthread_mutex_unlock(&stru_5260DA0);
        if ( pthread_mutex_trylock((pthread_mutex_t *)(v18 + 120)) )
        {
          sub_130AD90(v18 + 56);
          *(_BYTE *)(v18 + 160) = 1;
        }
        ++*(_QWORD *)(v18 + 112);
        if ( v2 != *(_BYTE **)(v18 + 104) )
        {
          ++*(_QWORD *)(v18 + 96);
          *(_QWORD *)(v18 + 104) = v2;
        }
      }
      else
      {
        if ( (unsigned int)v55 == unk_5260D40 )
          goto LABEL_43;
        *(_BYTE *)(v18 + 160) = 0;
        pthread_mutex_unlock((pthread_mutex_t *)(v18 + 120));
        if ( (unsigned __int64)*v62 <= 1 )
        {
          v43 = 0;
        }
        else
        {
          v60 = v2;
          v35 = 1;
          v36 = 1;
          while ( 1 )
          {
            v37 = &v57[v36];
            if ( !v57[v36] )
            {
              v38 = unk_5260DD8 + 208 * v36;
              v61 = (pthread_mutex_t *)(v38 + 120);
              v39 = pthread_mutex_trylock((pthread_mutex_t *)(v38 + 120));
              v40 = v61;
              if ( v39 )
              {
                sub_130AD90(v38 + 56);
                *(_BYTE *)(v38 + 160) = 1;
                v40 = v61;
              }
              ++*(_QWORD *)(v38 + 112);
              v41 = v60;
              if ( v60 != *(_BYTE **)(v38 + 104) )
              {
                ++*(_QWORD *)(v38 + 96);
                *(_QWORD *)(v38 + 104) = v41;
              }
              v42 = *(_DWORD *)(v38 + 168);
              *(_BYTE *)(v38 + 160) = 0;
              pthread_mutex_unlock(v40);
              if ( v42 == 1 )
                break;
            }
            v36 = ++v35;
            if ( v35 >= (unsigned __int64)*v62 )
            {
              v2 = v60;
              v43 = 0;
              goto LABEL_88;
            }
          }
          v46 = v36;
          v47 = v37;
          v2 = v60;
          ++v60[1];
          if ( !v2[816] )
            sub_1313A40(v2);
          v48 = sub_1319830(v38, v46);
          v49 = v2[1]-- == 1;
          v50 = v48;
          if ( v49 )
            sub_1313A40(v2);
          if ( v50 )
          {
            sub_130ACF0("<jemalloc>: background thread creation failed (%d)\n", v50);
            if ( byte_4F969A5[0] )
              abort();
            v43 = 1;
          }
          else
          {
            *v47 = 1;
            v43 = 1;
            LODWORD(v55) = (_DWORD)v55 + 1;
          }
        }
LABEL_88:
        v44 = unk_5260DD8;
        v45 = unk_5260DD8 + 56LL;
        if ( pthread_mutex_trylock((pthread_mutex_t *)(unk_5260DD8 + 120LL)) )
        {
          sub_130AD90(v45);
          *(_BYTE *)(v44 + 160) = 1;
        }
        ++*(_QWORD *)(v44 + 112);
        if ( v2 != *(_BYTE **)(v44 + 104) )
        {
          ++*(_QWORD *)(v44 + 96);
          *(_QWORD *)(v44 + 104) = v2;
        }
        if ( !v43 )
        {
LABEL_43:
          v22 = unk_5260DD8;
          v54 = (__int64 *)unk_5260DD8;
          v23 = sub_1300B70();
          LOBYTE(v61) = *(_BYTE *)(v22 + 172);
          if ( v23 )
          {
            v24 = (__int64)v2;
            v25 = v58;
            v26 = -1;
            do
            {
              v28 = qword_50579C0[v25];
              if ( v28 )
              {
                if ( !(_BYTE)v61 )
                {
                  v60 = (_BYTE *)qword_50579C0[v25];
                  sub_1315270(v24, v28);
                  v28 = (__int64)v60;
                }
                if ( v26 > 0x5F5E100 )
                {
                  v27 = sub_130B900(v24, v28 + 10648);
                  if ( v26 > v27 )
                    v26 = v27;
                }
              }
              v25 += *(_DWORD *)v62;
            }
            while ( v23 > v25 );
            v29 = 100000000;
            v2 = (_BYTE *)v24;
            if ( v26 >= 0x5F5E100 )
              v29 = v26;
          }
          else
          {
            v29 = -1;
          }
          sub_1319550((__int64)v54, v29);
        }
      }
      v18 = unk_5260DD8;
      v21 = *(_DWORD *)(unk_5260DD8 + 168LL);
    }
    while ( v21 );
    if ( (unsigned __int64)*v62 <= 1 )
      goto LABEL_69;
    goto LABEL_57;
  }
  owner = *(_DWORD *)(v5 + 168);
  v55 = (pthread_mutex_t *)(v5 + 160);
  if ( owner )
  {
    while ( owner != 2 )
    {
      LODWORD(v62) = sub_1300B70();
      LOBYTE(v61) = *v57;
      if ( v58 >= (unsigned int)v62 )
      {
        v13 = -1;
      }
      else
      {
        v8 = (__int64)v2;
        v9 = v58;
        v10 = -1;
        do
        {
          v12 = qword_50579C0[v9];
          if ( v12 )
          {
            if ( !(_BYTE)v61 )
            {
              v60 = (_BYTE *)qword_50579C0[v9];
              sub_1315270(v8, v12);
            }
            if ( v10 > 0x5F5E100 )
            {
              v11 = sub_130B900(v8, v12 + 10648);
              if ( v10 > v11 )
                v10 = v11;
            }
          }
          v9 += LODWORD(qword_5260D48[0]);
        }
        while ( (unsigned int)v62 > v9 );
        v13 = 100000000;
        v2 = (_BYTE *)v8;
        if ( v10 >= 0x5F5E100 )
          v13 = v10;
      }
      sub_1319550((__int64)v59, v13);
LABEL_23:
      owner = v59[4].__owner;
      if ( !owner )
        goto LABEL_24;
    }
    v55->__size[0] = 0;
    pthread_mutex_unlock(mutex);
    if ( pthread_mutex_trylock(&stru_5260DA0) )
    {
      sub_130AD90((__int64)&unk_5260D60);
      unk_5260DC8 = 1;
      ++unk_5260D98;
      if ( v2 == (_BYTE *)unk_5260D90 )
        goto LABEL_29;
    }
    else
    {
      ++unk_5260D98;
      if ( v2 == (_BYTE *)unk_5260D90 )
      {
LABEL_29:
        unk_5260DC8 = 0;
        pthread_mutex_unlock(&stru_5260DA0);
        if ( pthread_mutex_trylock(mutex) )
        {
          sub_130AD90((__int64)v54);
          v55->__size[0] = 1;
        }
        v15 = (__int64)v59;
        ++v59[2].__list.__next;
        if ( v2 != *(_BYTE **)(v15 + 104) )
        {
          ++*(_QWORD *)(v15 + 96);
          *(_QWORD *)(v15 + 104) = v2;
        }
        goto LABEL_23;
      }
    }
    ++unk_5260D88;
    unk_5260D90 = v2;
    goto LABEL_29;
  }
LABEL_24:
  v59[4].__size[12] = 0;
  sub_130B0C0(v53, 0);
  v55->__size[0] = 0;
  pthread_mutex_unlock(mutex);
  return 0;
}
