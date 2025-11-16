// Function: sub_130D630
// Address: 0x130d630
//
__int64 __fastcall sub_130D630(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  int v8; // eax
  unsigned __int64 v9; // r11
  __int64 v10; // r15
  unsigned __int64 v11; // r9
  int *v12; // r8
  __int64 v13; // rax
  __int64 v14; // r10
  unsigned __int64 v15; // r9
  __int64 v17; // rax
  __int64 v18; // r8
  pthread_mutex_t *mutex; // [rsp+10h] [rbp-50h]
  unsigned __int64 v21; // [rsp+18h] [rbp-48h]
  unsigned __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  char v25[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v21 = a5 + 4096;
  mutex = (pthread_mutex_t *)(a2 + 64);
  v8 = pthread_mutex_trylock((pthread_mutex_t *)(a2 + 64));
  v9 = v21;
  if ( v8 )
  {
    sub_130AD90(a2);
    *(_BYTE *)(a2 + 104) = 1;
    v9 = v21;
  }
  ++*(_QWORD *)(a2 + 56);
  if ( a1 != *(_QWORD *)(a2 + 48) )
  {
    ++*(_QWORD *)(a2 + 40);
    *(_QWORD *)(a2 + 48) = a1;
  }
  v10 = *(_QWORD *)(a2 + 112);
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 16) & 0xFFFFFFFFFFFFF000LL;
    if ( v11 >= v9 )
    {
      v14 = 0;
      v15 = v11 - v9;
      if ( !v15 )
        goto LABEL_11;
      goto LABEL_18;
    }
  }
  v12 = &dword_400000;
  v25[0] = 0;
  if ( v9 >= (unsigned __int64)&dword_400000 )
    LODWORD(v12) = v9;
  v22 = v9;
  v13 = sub_1344390(a1, a3, a4, 0, (_DWORD)v12, 4096, 0, (__int64)v25);
  *(_QWORD *)(a2 + 112) = v13;
  if ( !v13 )
    goto LABEL_20;
  v9 = v22;
  v14 = v10;
  v10 = v13;
  v15 = (*(_QWORD *)(v13 + 16) & 0xFFFFFFFFFFFFF000LL) - v22;
  if ( v15 )
  {
LABEL_18:
    v24 = v14;
    v17 = sub_13457A0(a1, a3, a4, v10, v9, v15, 1);
    if ( v17 )
    {
      v10 = *(_QWORD *)(a2 + 112);
      v14 = v24;
      *(_QWORD *)(a2 + 112) = v17;
      goto LABEL_12;
    }
LABEL_20:
    *(_BYTE *)(a2 + 104) = 0;
    v10 = 0;
    pthread_mutex_unlock(mutex);
    return v10;
  }
LABEL_11:
  *(_QWORD *)(a2 + 112) = 0;
LABEL_12:
  v23 = v14;
  *(_BYTE *)(a2 + 104) = 0;
  pthread_mutex_unlock(mutex);
  if ( v23 )
    sub_13446B0(a1, a3, a4, v23);
  sub_130D840(a1, a4, v10, *(_QWORD *)(a3 + 58384), 0, 1, 1);
  if ( (unsigned __int8)sub_13457C0(a1, a4, v10, 1, a6, 0) )
  {
    v18 = v10;
    v10 = 0;
    sub_13451C0(a1, a3, a4, a3 + 38936, v18);
  }
  return v10;
}
