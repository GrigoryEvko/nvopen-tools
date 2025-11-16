// Function: sub_1317CF0
// Address: 0x1317cf0
//
__int64 __fastcall sub_1317CF0(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4, unsigned __int8 a5)
{
  __int64 v7; // r12
  __int64 v8; // rax
  pthread_mutex_t *v9; // r14
  __int64 v10; // rbx
  __int64 v11; // r8
  void *v12; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 *v17; // [rsp+8h] [rbp-68h]
  void *sb; // [rsp+10h] [rbp-60h]
  _QWORD *s; // [rsp+10h] [rbp-60h]
  void *sa; // [rsp+10h] [rbp-60h]
  size_t n; // [rsp+18h] [rbp-58h]
  _BYTE *v23; // [rsp+28h] [rbp-48h]
  void *v24; // [rsp+28h] [rbp-48h]
  unsigned int v25[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v7 = a2;
  if ( a1 )
  {
    if ( a2 )
      goto LABEL_3;
    if ( unk_4C6F220 > a3 || (v16 = *(_QWORD *)(a1 + 144)) != 0 && *(_DWORD *)(v16 + 78928) >= dword_5057900[0] )
      v7 = sub_1314520(a1, 0);
    else
      v7 = sub_1317C00(a1);
  }
  if ( !v7 )
    return 0;
LABEL_3:
  n = qword_505FA40[a4];
  if ( a3 <= 0x3800 )
  {
    v8 = sub_1315920(a1, v7, a4, v25);
    v9 = (pthread_mutex_t *)(v8 + 64);
    v10 = v8;
    if ( pthread_mutex_trylock((pthread_mutex_t *)(v8 + 64)) )
    {
      sub_130AD90(v10);
      v23 = (_BYTE *)(v10 + 104);
      *(_BYTE *)(v10 + 104) = 1;
    }
    else
    {
      v23 = (_BYTE *)(v10 + 104);
    }
    ++*(_QWORD *)(v10 + 56);
    if ( a1 != *(_QWORD *)(v10 + 48) )
    {
      ++*(_QWORD *)(v10 + 40);
      *(_QWORD *)(v10 + 48) = a1;
    }
    v11 = sub_1314420(v7, (_QWORD *)v10, a4);
    if ( v11 )
      goto LABEL_9;
    s = (_QWORD *)((char *)&unk_5260DE0 + 40 * a4);
    *v23 = 0;
    pthread_mutex_unlock(v9);
    v17 = sub_1316650(a1, v7, a4, v25[0], (__int64)s);
    if ( pthread_mutex_trylock(v9) )
    {
      sub_130AD90(v10);
      *v23 = 1;
    }
    ++*(_QWORD *)(v10 + 56);
    if ( a1 != *(_QWORD *)(v10 + 48) )
    {
      ++*(_QWORD *)(v10 + 40);
      *(_QWORD *)(v10 + 48) = a1;
    }
    v15 = sub_1314420(v7, (_QWORD *)v10, a4);
    if ( v15 )
    {
      sa = (void *)v15;
      ++*(_QWORD *)(v10 + 112);
      ++*(_QWORD *)(v10 + 128);
      ++*(_QWORD *)(v10 + 136);
      *v23 = 0;
      pthread_mutex_unlock(v9);
      v12 = sa;
      if ( v17 )
      {
        sub_13152A0(a1, v7, v17);
        v12 = sa;
      }
      goto LABEL_10;
    }
    if ( v17 )
    {
      ++*(_QWORD *)(v10 + 160);
      ++*(_QWORD *)(v10 + 176);
      *(_QWORD *)(v10 + 192) = v17;
      v11 = sub_13143B0(v17, s);
LABEL_9:
      sb = (void *)v11;
      ++*(_QWORD *)(v10 + 112);
      ++*(_QWORD *)(v10 + 128);
      ++*(_QWORD *)(v10 + 136);
      *v23 = 0;
      pthread_mutex_unlock(v9);
      v12 = sb;
LABEL_10:
      if ( a5 )
        v12 = memset(v12, 0, n);
      if ( a1 )
      {
        if ( --*(_DWORD *)(a1 + 152) < 0 )
        {
          if ( (unsigned __int8)sub_1314130((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112)) )
          {
            v24 = v12;
            sub_1315160(a1, v7, 0, 0);
            return (__int64)v24;
          }
        }
      }
      return (__int64)v12;
    }
    *v23 = 0;
    pthread_mutex_unlock(v9);
    return 0;
  }
  return sub_1309DC0(a1, v7, n, a5);
}
