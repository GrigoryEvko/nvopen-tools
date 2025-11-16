// Function: sub_130C320
// Address: 0x130c320
//
__int64 __fastcall sub_130C320(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        volatile signed __int64 *a4,
        __int64 a5,
        char a6,
        __int64 a7,
        unsigned __int64 a8)
{
  __int64 v8; // rax
  __int64 v9; // r13
  _QWORD *v10; // r14
  unsigned __int64 v11; // r15
  _QWORD *v12; // rax
  __int64 v14; // r15
  _QWORD *v15; // r9
  __int64 v16; // r14
  _QWORD *v17; // r13
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // r8
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  char v23; // al
  _QWORD *v24; // [rsp+8h] [rbp-78h]
  _BOOL4 v25; // [rsp+14h] [rbp-6Ch]
  pthread_mutex_t *mutex; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  unsigned __int64 v29; // [rsp+30h] [rbp-50h]
  unsigned __int64 v31; // [rsp+38h] [rbp-48h]

  *(_BYTE *)(a3 + 112) = 1;
  *(_BYTE *)(a3 + 104) = 0;
  mutex = (pthread_mutex_t *)(a3 + 64);
  pthread_mutex_unlock((pthread_mutex_t *)(a3 + 64));
  v8 = sub_131C0E0(*(_QWORD *)(a2 + 58376));
  if ( !a8 )
  {
LABEL_9:
    if ( !pthread_mutex_trylock(mutex) )
      goto LABEL_10;
    goto LABEL_25;
  }
  v9 = v8;
  v10 = 0;
  v11 = 0;
  while ( 1 )
  {
    v12 = (_QWORD *)sub_13441C0(a1, a2, v9, a5, a7);
    if ( !v12 )
      break;
    v12[8] = v12;
    v12[9] = v12;
    if ( v10 )
    {
      v12[8] = v10[9];
      v10[9] = v12;
      v12[9] = *(_QWORD *)(v12[9] + 64LL);
      *(_QWORD *)(v10[9] + 64LL) = v10;
      *(_QWORD *)(v12[9] + 64LL) = v12;
      v10 = (_QWORD *)v12[8];
    }
    else
    {
      v10 = v12;
    }
    v11 += v12[2] >> 12;
    if ( a8 <= v11 )
      goto LABEL_13;
  }
  if ( !v11 )
    goto LABEL_9;
LABEL_13:
  v25 = 0;
  v14 = sub_131C0E0(*(_QWORD *)(a2 + 58376));
  if ( !a6 )
    v25 = sub_130C0D0(a2, 2) != 0;
  if ( v10 )
  {
    v15 = v10;
    v28 = 0;
    v16 = v14;
    v29 = 0;
    v31 = 0;
    while ( 1 )
    {
      v17 = (_QWORD *)v15[8];
      if ( v17 == v15 )
      {
        v17 = 0;
      }
      else
      {
        *(_QWORD *)(v15[9] + 64LL) = v17[9];
        v22 = v15[9];
        *(_QWORD *)(v15[8] + 72LL) = v22;
        v15[9] = *(_QWORD *)(v22 + 64);
        *(_QWORD *)(*(_QWORD *)(v15[8] + 72LL) + 64LL) = v15[8];
        *(_QWORD *)(v15[9] + 64LL) = v15;
      }
      ++v31;
      v18 = v15[2];
      v29 += v18 >> 12;
      v19 = v18 >> 12;
      v20 = v18 & 0xFFFFFFFFFFFFF000LL;
      if ( *(_DWORD *)(a5 + 19424) != 1 )
        goto LABEL_21;
      if ( v25 && (v24 = v15, v23 = sub_1345440(a1, v16, v15, 0, v20), v15 = v24, !v23) )
      {
        sub_13453B0(a1, a2, v16, a2 + 19496, v24);
      }
      else
      {
LABEL_21:
        sub_1344AD0(a1, a2, v16, v15);
        v28 += v19;
      }
      if ( !v17 )
        break;
      v15 = v17;
    }
    v21 = v28 << 12;
  }
  else
  {
    v29 = 0;
    v21 = 0;
    v31 = 0;
  }
  _InterlockedAdd64(a4, 1u);
  _InterlockedAdd64(a4 + 1, v31);
  _InterlockedAdd64(a4 + 2, v29);
  _InterlockedSub64((volatile signed __int64 *)(*(_QWORD *)(a2 + 62224) + 56LL), v21);
  if ( pthread_mutex_trylock(mutex) )
  {
LABEL_25:
    sub_130AD90(a3);
    *(_BYTE *)(a3 + 104) = 1;
  }
LABEL_10:
  ++*(_QWORD *)(a3 + 56);
  if ( a1 != *(_QWORD *)(a3 + 48) )
  {
    ++*(_QWORD *)(a3 + 40);
    *(_QWORD *)(a3 + 48) = a1;
  }
  *(_BYTE *)(a3 + 112) = 0;
  return a3;
}
