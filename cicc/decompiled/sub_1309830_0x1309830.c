// Function: sub_1309830
// Address: 0x1309830
//
__int64 __fastcall sub_1309830(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, unsigned __int8 a5)
{
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  _QWORD *v10; // r14
  __int64 v13; // rcx
  int v14; // eax
  unsigned int v15; // edx
  unsigned int v16; // eax
  __int64 v17; // rbx
  int v18; // eax
  _BYTE *v19; // rcx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int8 v30; // [rsp+8h] [rbp-38h]
  unsigned __int8 v31; // [rsp+8h] [rbp-38h]
  unsigned __int8 v32; // [rsp+8h] [rbp-38h]
  unsigned __int8 v33; // [rsp+8h] [rbp-38h]
  unsigned __int8 v34; // [rsp+8h] [rbp-38h]

  v7 = a2;
  if ( a4 <= 0x1000 && a3 <= 0x3800 )
  {
    v8 = -(__int64)a4 & (a3 + a4 - 1);
    if ( v8 > 0x1000 )
    {
      if ( v8 > 0x7000000000000000LL )
        return 0;
      _BitScanReverse64(&v23, 2 * v8 - 1);
      v9 = -(1LL << ((unsigned __int8)v23 - 3)) & (v8 + (1LL << ((unsigned __int8)v23 - 3)) - 1);
    }
    else
    {
      v9 = qword_505FA40[byte_5060800[(v8 + 7) >> 3]];
    }
    if ( v9 > 0x3FFF )
      goto LABEL_6;
LABEL_21:
    if ( v9 - 1 > 0x6FFFFFFFFFFFFFFFLL )
      return 0;
    goto LABEL_7;
  }
  if ( a4 > 0x7000000000000000LL )
    return 0;
  if ( a3 > 0x4000 )
  {
    if ( a3 > 0x7000000000000000LL )
      return 0;
    _BitScanReverse64((unsigned __int64 *)&v13, 2 * a3 - 1);
    if ( (unsigned __int64)(int)v13 < 7 )
      LOBYTE(v13) = 7;
    v9 = -(1LL << ((unsigned __int8)v13 - 3)) & ((1LL << ((unsigned __int8)v13 - 3)) + a3 - 1);
    if ( a3 > v9 || __CFADD__(v9, ((a4 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 - 4096) )
      return 0;
    goto LABEL_21;
  }
LABEL_6:
  if ( ((a4 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 + 12288 <= 0x3FFF )
    return 0;
LABEL_7:
  if ( !a1 )
    goto LABEL_50;
  if ( a2 )
    goto LABEL_9;
  if ( a3 >= unk_4C6F220 )
  {
    v7 = *(_QWORD *)(a1 + 144);
    if ( !v7 || *(_DWORD *)(v7 + 78928) < unk_5057900 )
    {
      v32 = a5;
      v22 = sub_1317C00(a1);
      a5 = v32;
      v7 = v22;
      goto LABEL_50;
    }
    if ( *(char *)(a1 + 1) <= 0 )
      goto LABEL_26;
LABEL_59:
    v7 = qword_50579C0[0];
    if ( qword_50579C0[0] )
      goto LABEL_9;
    v33 = a5;
    v24 = sub_1300B80(a1, 0, (__int64)&off_49E8000);
    a5 = v33;
    v7 = v24;
    goto LABEL_50;
  }
  if ( *(char *)(a1 + 1) > 0 )
    goto LABEL_59;
  v7 = *(_QWORD *)(a1 + 144);
  if ( v7 )
    goto LABEL_26;
  v34 = a5;
  v25 = sub_1302AE0(a1, 0);
  a5 = v34;
  v7 = v25;
  if ( !*(_BYTE *)a1 )
    goto LABEL_65;
  v26 = *(_QWORD *)(a1 + 296);
  v27 = a1 + 256;
  v28 = a1 + 856;
  if ( v26 )
  {
    if ( v7 != v26 )
    {
      sub_1311F50(a1, v27, v28, v7);
      a5 = v34;
      goto LABEL_65;
    }
LABEL_26:
    v14 = unk_4C6F238;
    if ( unk_4C6F238 <= 2u )
      goto LABEL_9;
    goto LABEL_27;
  }
  sub_13114E0(a1, v27, v28, v7);
  a5 = v34;
LABEL_65:
  v14 = unk_4C6F238;
  if ( unk_4C6F238 > 2u )
  {
LABEL_27:
    v15 = dword_505F9BC;
    if ( v14 == 4 && dword_505F9BC > 1u )
      v15 = (dword_505F9BC >> 1) - (((dword_505F9BC & 1) == 0) - 1);
    if ( *(_DWORD *)(v7 + 78928) < v15 && a1 != *(_QWORD *)(v7 + 16) )
    {
      v30 = a5;
      v16 = sched_getcpu();
      a5 = v30;
      if ( unk_4C6F238 != 3 && dword_505F9BC >> 1 <= v16 )
        v16 -= dword_505F9BC >> 1;
      if ( *(_DWORD *)(v7 + 78928) != v16 )
      {
        v7 = *(_QWORD *)(a1 + 144);
        if ( v16 != *(_DWORD *)(v7 + 78928) )
        {
          v17 = qword_50579C0[v16];
          if ( !v17 )
          {
            v29 = sub_1300B80(a1, v16, (__int64)&off_49E8000);
            a5 = v30;
            v17 = v29;
          }
          v31 = a5;
          sub_1302A70(a1, v7, v17);
          a5 = v31;
          if ( *(_BYTE *)a1 )
          {
            sub_1311F50(a1, a1 + 256, a1 + 856, v17);
            v7 = *(_QWORD *)(a1 + 144);
            a5 = v31;
          }
          else
          {
            v7 = *(_QWORD *)(a1 + 144);
          }
        }
      }
      *(_QWORD *)(v7 + 16) = a1;
    }
    goto LABEL_9;
  }
LABEL_50:
  if ( !v7 )
    return 0;
LABEL_9:
  v10 = (_QWORD *)sub_1316380(a1, v7, a3, a4, a5);
  if ( !v10 )
    return 0;
  if ( *(_DWORD *)(v7 + 78928) >= unk_5057900 )
  {
    v18 = pthread_mutex_trylock((pthread_mutex_t *)(v7 + 10600));
    v19 = (_BYTE *)(v7 + 10640);
    if ( v18 )
    {
      sub_130AD90(v7 + 10536);
      *(_BYTE *)(v7 + 10640) = 1;
      v19 = (_BYTE *)(v7 + 10640);
    }
    ++*(_QWORD *)(v7 + 10592);
    if ( a1 != *(_QWORD *)(v7 + 10584) )
    {
      ++*(_QWORD *)(v7 + 10576);
      *(_QWORD *)(v7 + 10584) = a1;
    }
    v10[5] = v10;
    v20 = v10;
    v10[6] = v10;
    v21 = *(_QWORD *)(v7 + 10528);
    if ( v21 )
    {
      v10[5] = *(_QWORD *)(v21 + 48);
      *(_QWORD *)(*(_QWORD *)(v7 + 10528) + 48LL) = v10;
      v10[6] = *(_QWORD *)(v10[6] + 40LL);
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 10528) + 48LL) + 40LL) = *(_QWORD *)(v7 + 10528);
      *(_QWORD *)(v10[6] + 40LL) = v10;
      v20 = (_QWORD *)v10[5];
    }
    *(_QWORD *)(v7 + 10528) = v20;
    *v19 = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(v7 + 10600));
  }
  if ( a1 )
  {
    if ( --*(_DWORD *)(a1 + 152) < 0 )
    {
      if ( (unsigned __int8)sub_1309470((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112)) )
        sub_1315160(a1, v7, 0, 0);
    }
  }
  return v10[1];
}
