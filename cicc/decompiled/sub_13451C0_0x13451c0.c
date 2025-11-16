// Function: sub_13451C0
// Address: 0x13451c0
//
int __fastcall sub_13451C0(_BYTE *a1, __int64 *a2, unsigned int *a3, __int64 a4, __int64 *a5)
{
  char *v8; // r9
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v12; // rbx
  pthread_mutex_t *mutex; // [rsp+0h] [rbp-50h]
  char *v15; // [rsp+8h] [rbp-48h]
  char v16; // [rsp+1Fh] [rbp-31h] BYREF

  mutex = (pthread_mutex_t *)(a4 + 64);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a4 + 64)) )
  {
    sub_130AD90(a4);
    *(_BYTE *)(a4 + 104) = 1;
  }
  ++*(_QWORD *)(a4 + 56);
  if ( a1 != *(_BYTE **)(a4 + 48) )
  {
    ++*(_QWORD *)(a4 + 40);
    *(_QWORD *)(a4 + 48) = a1;
  }
  if ( (*a5 & 0x10000) == 0 )
  {
    if ( *(_BYTE *)(a4 + 19432) )
    {
      if ( (a5[2] & 0xFFFFFFFFFFFFF000LL) > 0x3FFF )
      {
        v8 = &v16;
        do
        {
          v15 = v8;
          v9 = sub_13436B0(a1, (__int64)a2, a3, a4, a5, v8);
          v8 = v15;
          a5 = v9;
        }
        while ( v16 );
        if ( a2[7330] <= (v9[2] & 0xFFFFFFFFFFFFF000LL)
          && sub_130C0D0((__int64)a2, 1) != -1
          && sub_130C0D0((__int64)a2, 2) != -1 )
        {
          *(_BYTE *)(a4 + 104) = 0;
          pthread_mutex_unlock(mutex);
          v12 = a5[2];
          sub_1344AD0(a1, (__int64)a2, a3, (unsigned __int64 *)a5);
          v12 &= 0xFFFFFFFFFFFFF000LL;
          _InterlockedAdd64((volatile signed __int64 *)(a2[7778] + 8), 1u);
          _InterlockedAdd64((volatile signed __int64 *)(a2[7778] + 16), v12 >> 12);
          v11 = a2[7778];
          _InterlockedSub64((volatile signed __int64 *)(v11 + 56), v12);
          return v11;
        }
      }
    }
    else
    {
      a5 = sub_13436B0(a1, (__int64)a2, a3, a4, a5, 0);
    }
  }
  sub_1341570((__int64)a1, a2[7298], (unsigned __int64 *)a5, *(_DWORD *)(a4 + 19424));
  v10 = a4 + 112;
  if ( (*a5 & 0x10000) != 0 )
    v10 = a4 + 9768;
  sub_1342830(v10, a5);
  *(_BYTE *)(a4 + 104) = 0;
  LODWORD(v11) = pthread_mutex_unlock(mutex);
  return v11;
}
