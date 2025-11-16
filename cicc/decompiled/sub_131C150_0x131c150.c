// Function: sub_131C150
// Address: 0x131c150
//
_QWORD *__fastcall sub_131C150(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  pthread_mutex_t *v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v11; // r13
  unsigned __int64 *v12; // r9
  __int64 v13; // rax
  __int64 v14; // r9
  _QWORD *v15; // r13
  __int64 v16; // r10
  __int64 v18; // rax
  __int64 (__fastcall ***v19)(int, int, int, int, int, int, int); // r13
  char v20; // cl
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-60h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = (pthread_mutex_t *)(a2 + 96);
  v6 = (a4 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  v23 = -(__int64)v6 & (v6 + a3 - 1);
  v7 = v6 + v23;
  v8 = v6 + v23 - 16;
  v22 = a2 + 32;
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 96)) )
  {
    sub_130AD90(v22);
    *(_BYTE *)(a2 + 136) = 1;
  }
  ++*(_QWORD *)(a2 + 88);
  if ( a1 != *(_BYTE **)(a2 + 80) )
  {
    ++*(_QWORD *)(a2 + 72);
    *(_QWORD *)(a2 + 80) = a1;
  }
  if ( v8 > 0x1000 )
  {
    if ( v8 > 0x7000000000000000LL )
      goto LABEL_14;
    v20 = 7;
    _BitScanReverse64((unsigned __int64 *)&v21, 2 * v8 - 1);
    if ( (unsigned int)v21 >= 7 )
      v20 = v21;
    if ( (unsigned int)v21 < 6 )
      LODWORD(v21) = 6;
    v9 = ((unsigned int)(((-1LL << (v20 - 3)) & (v7 - 17)) >> (v20 - 3)) & 3) + 4 * (_DWORD)v21 - 23;
  }
  else
  {
    v9 = byte_5060800[(v7 - 9) >> 3];
  }
  if ( (unsigned int)v9 <= 0xE7 )
  {
    v10 = a2 + 16LL * (unsigned int)(v9 + 10) + 8;
    v11 = a2 + 16 * (v9 + (unsigned int)(231 - v9)) + 184;
    do
    {
      v12 = (unsigned __int64 *)sub_133FAA0(v10);
      if ( v12 )
        goto LABEL_11;
      v10 += 16;
    }
    while ( v11 != v10 );
  }
LABEL_14:
  v18 = sub_131C0F0(a2);
  *(_BYTE *)(a2 + 136) = 0;
  v19 = (__int64 (__fastcall ***)(int, int, int, int, int, int, int))v18;
  pthread_mutex_unlock(v5);
  v15 = sub_131B220(a1, a2, v19, (int *)(a2 + 148), (_QWORD *)(a2 + 152), v23, v6);
  if ( pthread_mutex_trylock(v5) )
  {
    sub_130AD90(v22);
    *(_BYTE *)(a2 + 136) = 1;
  }
  ++*(_QWORD *)(a2 + 88);
  if ( a1 != *(_BYTE **)(a2 + 80) )
  {
    ++*(_QWORD *)(a2 + 72);
    *(_QWORD *)(a2 + 80) = a1;
  }
  if ( v15 )
  {
    v15[1] = *(_QWORD *)(a2 + 160);
    *(_QWORD *)(a2 + 160) = v15;
    *(_QWORD *)(a2 + 3880) += 144LL;
    *(_QWORD *)(a2 + 3888) += 4096LL;
    *(_QWORD *)(a2 + 3896) += *v15;
    if ( dword_4F96B94 && !unk_505F9C8 && (dword_4F96B94 != 1 || *(_BYTE *)(a2 + 144)) )
      ++*(_QWORD *)(a2 + 3904);
    v12 = v15 + 2;
LABEL_11:
    v13 = sub_131B1D0(v12, v26, v23, v6);
    v25 = v14;
    v15 = (_QWORD *)v13;
    sub_131B080(a2, v14, v26[0], v13, v16);
    if ( a5 )
      *a5 = *(_QWORD *)(v25 + 32);
  }
  *(_BYTE *)(a2 + 136) = 0;
  pthread_mutex_unlock(v5);
  return v15;
}
