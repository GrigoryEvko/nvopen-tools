// Function: sub_130D370
// Address: 0x130d370
//
unsigned __int64 __fastcall sub_130D370(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4, char a5, char a6)
{
  unsigned __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // rbx
  unsigned __int64 *v12; // rax
  int v14; // eax
  pthread_mutex_t *v15; // r8
  __int64 *v16; // r11
  __int64 v17; // r10
  __int64 v18; // rax
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 *v20; // [rsp+10h] [rbp-40h]
  pthread_mutex_t *mutex; // [rsp+18h] [rbp-38h]

  v8 = a4 >> 30;
  v9 = (a4 >> 30) & 0x3FFFF;
  v10 = *(_QWORD *)(a2 + 8 * v9 + 120);
  if ( a6 )
  {
    if ( !a5 && !v10 )
    {
      v19 = a2 + 8 * v9 + 120;
      v14 = pthread_mutex_trylock((pthread_mutex_t *)(a2 + 72));
      v15 = (pthread_mutex_t *)(a2 + 72);
      v16 = (__int64 *)v19;
      v17 = a1;
      if ( v14 )
      {
        sub_130AD90(a2 + 8);
        *(_BYTE *)(a2 + 112) = 1;
        v17 = a1;
        v16 = (__int64 *)v19;
        v15 = (pthread_mutex_t *)(a2 + 72);
      }
      ++*(_QWORD *)(a2 + 64);
      if ( v17 != *(_QWORD *)(a2 + 56) )
      {
        ++*(_QWORD *)(a2 + 48);
        *(_QWORD *)(a2 + 56) = v17;
      }
      v10 = *v16;
      if ( !*v16 )
      {
        v20 = v16;
        mutex = v15;
        v18 = sub_131C440(v17, *(_QWORD *)a2, 0x200000, 64);
        v15 = mutex;
        v10 = v18;
        if ( !v18 )
        {
          *(_BYTE *)(a2 + 112) = 0;
          pthread_mutex_unlock(mutex);
          return v10;
        }
        *v20 = v18;
      }
      *(_BYTE *)(a2 + 112) = 0;
      pthread_mutex_unlock(v15);
    }
    goto LABEL_3;
  }
  if ( a5 || v10 )
  {
LABEL_3:
    v11 = 2 * (v8 & 0xF);
    memmove(a3 + 34, a3 + 32, 0x70u);
    v12 = &a3[v11];
    a3[32] = a3[v11];
    a3[33] = a3[v11 + 1];
    v12[1] = v10;
    *v12 = a4 & 0xFFFFFFFFC0000000LL;
    v10 += (a4 >> 9) & 0x1FFFF8;
  }
  return v10;
}
