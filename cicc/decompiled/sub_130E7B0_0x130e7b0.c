// Function: sub_130E7B0
// Address: 0x130e7b0
//
int __fastcall sub_130E7B0(__int64 a1, _QWORD *a2, _QWORD *a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // r14
  int v9; // eax
  pthread_mutex_t *v10; // r8
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned int v13; // edx
  char v14; // cl
  unsigned int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rdx

  if ( !a2[8] || a2[9] < (a3[2] & 0xFFFFFFFFFFFFF000LL) )
    return (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD *, __int64))(a2[7] + 32LL))(a1, a2[7], a3, a4);
  if ( a1 )
  {
    v7 = *(unsigned __int8 *)(a1 + 160);
    if ( (_BYTE)v7 == 0xFF )
    {
      v20 = 0x5851F42D4C957F2DLL * *(_QWORD *)(a1 + 112) + 0x14057B7EF767814FLL;
      *(_QWORD *)(a1 + 112) = v20;
      v21 = (a2[8] * HIDWORD(v20)) >> 32;
      *(_BYTE *)(a1 + 160) = v21;
      v7 = (unsigned __int8)v21;
    }
    v8 = a2[13] + 144 * v7;
  }
  else
  {
    v8 = a2[13];
  }
  v9 = pthread_mutex_trylock((pthread_mutex_t *)(v8 + 64));
  v10 = (pthread_mutex_t *)(v8 + 64);
  if ( v9 )
  {
    sub_130AD90(v8);
    *(_BYTE *)(v8 + 104) = 1;
    v10 = (pthread_mutex_t *)(v8 + 64);
  }
  ++*(_QWORD *)(v8 + 56);
  if ( a1 != *(_QWORD *)(v8 + 48) )
  {
    ++*(_QWORD *)(v8 + 40);
    *(_QWORD *)(v8 + 48) = a1;
  }
  if ( !*(_BYTE *)(v8 + 112) )
  {
    *(_BYTE *)(v8 + 104) = 0;
    pthread_mutex_unlock(v10);
    return (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD *, __int64))(a2[7] + 32LL))(a1, a2[7], a3, a4);
  }
  v11 = a3[2] & 0xFFFFFFFFFFFFF000LL;
  if ( v11 > 0x7000000000000000LL )
  {
    v16 = 4776;
  }
  else
  {
    _BitScanReverse64(&v12, v11);
    v13 = v12 - ((((v11 - 1) & v11) == 0) - 1);
    if ( v13 < 0xE )
      v13 = 14;
    v14 = v13 - 3;
    v15 = v13 - 14;
    if ( !v15 )
      v14 = 12;
    v16 = 24 * ((((v11 - 1) >> v14) & 3) + 4 * v15);
  }
  v17 = *(_QWORD *)(v8 + 120) + v16;
  a3[5] = a3;
  a3[6] = a3;
  v18 = *(_QWORD *)(v17 + 16);
  if ( v18 )
  {
    a3[5] = *(_QWORD *)(v18 + 48);
    *(_QWORD *)(*(_QWORD *)(v17 + 16) + 48LL) = a3;
    a3[6] = *(_QWORD *)(a3[6] + 40LL);
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v17 + 16) + 48LL) + 40LL) = *(_QWORD *)(v17 + 16);
    *(_QWORD *)(a3[6] + 40LL) = a3;
  }
  *(_QWORD *)(v17 + 8) += v11;
  *(_QWORD *)(v17 + 16) = a3;
  v19 = *(_QWORD *)(v8 + 128) + v11;
  *(_QWORD *)(v8 + 128) = v19;
  if ( v19 > a2[10] )
    return sub_130E170(a1, (__int64)a2, v8);
  *(_BYTE *)(v8 + 104) = 0;
  return pthread_mutex_unlock(v10);
}
