// Function: sub_130AD90
// Address: 0x130ad90
//
int __fastcall sub_130AD90(__int64 a1)
{
  pthread_mutex_t *v1; // r14
  __int64 v2; // rbx
  bool v3; // dl
  int result; // eax
  volatile signed __int32 *v5; // rbx
  signed __int32 v6; // [rsp+Ch] [rbp-54h]
  _BYTE v7[8]; // [rsp+18h] [rbp-48h] BYREF
  _BYTE v8[8]; // [rsp+20h] [rbp-40h] BYREF
  _BYTE v9[56]; // [rsp+28h] [rbp-38h] BYREF

  v1 = (pthread_mutex_t *)(a1 + 64);
  v2 = 0;
  if ( dword_505F9BC != 1 )
  {
    do
    {
      _mm_pause();
      if ( !*(_BYTE *)(a1 + 104) )
      {
        result = pthread_mutex_trylock(v1);
        if ( !result )
          goto LABEL_6;
      }
      v3 = qword_4C6F0E8 > v2++;
    }
    while ( qword_4C6F0E8 == -1 || v3 );
  }
  v5 = (volatile signed __int32 *)(a1 + 36);
  sub_130B270(v7);
  sub_130B140(v8, v7);
  v6 = _InterlockedExchangeAdd((volatile signed __int32 *)(a1 + 36), 1u);
  result = pthread_mutex_trylock(v1);
  if ( !result )
  {
    _InterlockedSub(v5, 1u);
LABEL_6:
    ++*(_QWORD *)(a1 + 24);
    return result;
  }
  pthread_mutex_lock(v1);
  *(_BYTE *)(a1 + 104) = 1;
  _InterlockedSub(v5, 1u);
  sub_130B160(v8);
  sub_130B140(v9, v8);
  sub_130B1F0(v9, v7);
  ++*(_QWORD *)(a1 + 16);
  sub_130B1D0(a1, v9);
  result = sub_130B150(a1 + 8, v9);
  if ( result < 0 )
    result = sub_130B140(a1 + 8, v9);
  if ( *(_DWORD *)(a1 + 32) < (unsigned int)(v6 + 1) )
    *(_DWORD *)(a1 + 32) = v6 + 1;
  return result;
}
