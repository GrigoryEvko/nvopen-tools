// Function: sub_1311650
// Address: 0x1311650
//
_DWORD *__fastcall sub_1311650(__int64 a1, __int64 a2, __int64 a3)
{
  _DWORD *result; // rax
  unsigned int v5; // r14d
  __int64 v6; // rax
  pthread_mutex_t *v7; // r15
  __int64 v8; // r12
  _BYTE *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-48h]

  v12 = a3 + 24;
  result = (_DWORD *)dword_5060A18[0];
  if ( dword_5060A18[0] )
  {
    v5 = 0;
    do
    {
      if ( v5 <= 0x23 )
      {
        v6 = sub_1315920(a1, a3, v5, 0);
        v7 = (pthread_mutex_t *)(v6 + 64);
        v8 = v6;
        v9 = (_BYTE *)(v6 + 104);
        if ( pthread_mutex_trylock((pthread_mutex_t *)(v6 + 64)) )
        {
          sub_130AD90(v8);
          v9 = (_BYTE *)(v8 + 104);
          *(_BYTE *)(v8 + 104) = 1;
        }
        ++*(_QWORD *)(v8 + 56);
        if ( a1 != *(_QWORD *)(v8 + 48) )
        {
          ++*(_QWORD *)(v8 + 40);
          *(_QWORD *)(v8 + 48) = a1;
        }
        *(_QWORD *)(v8 + 128) += *(_QWORD *)(a2 + 24LL * v5 + 16);
        *v9 = 0;
        pthread_mutex_unlock(v7);
        v10 = v5;
      }
      else
      {
        v10 = v5;
        v11 = 16 * (3LL * v5 - 108);
        _InterlockedAdd64((volatile signed __int64 *)(v12 + v11 + 968), *(_QWORD *)(a2 + 24LL * v5 + 16));
        _InterlockedAdd64((volatile signed __int64 *)(v12 + v11 + 984), 1u);
      }
      ++v5;
      *(_QWORD *)(a2 + 24 * v10 + 16) = 0;
      result = dword_5060A18;
    }
    while ( dword_5060A18[0] > v5 );
  }
  return result;
}
