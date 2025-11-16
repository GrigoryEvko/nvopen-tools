// Function: sub_13022D0
// Address: 0x13022d0
//
__int64 __fastcall sub_13022D0(__int64 a1, __int64 a2)
{
  pthread_mutex_t *v2; // rdi
  unsigned int v3; // r12d
  unsigned int v4; // r13d
  __int64 v5; // r12
  int v6; // eax
  _DWORD *v7; // rcx
  __int64 v8; // r15
  unsigned int *v9; // rdx
  unsigned __int64 v10; // rcx
  int v11; // r8d
  int v12; // r9d
  unsigned int v13; // eax
  unsigned int v14; // edx
  bool v15; // zf
  __int64 v16; // rsi
  int v17; // eax
  unsigned int v18; // r12d
  unsigned int v20; // esi
  int v21; // edx
  int v22; // r8d
  int v23; // r9d
  char v24; // di
  char v25; // si
  char v26; // cl
  unsigned __int8 v27; // dl
  unsigned __int64 v28; // r12
  unsigned int v29; // esi
  cpu_set_t cpuset; // [rsp+0h] [rbp-B0h] BYREF

  v2 = &stru_4F96940;
  if ( pthread_mutex_trylock(&stru_4F96940) )
  {
    v2 = (pthread_mutex_t *)&unk_4F96900;
    sub_130AD90(&unk_4F96900);
    byte_4F96968 = 1;
  }
  ++qword_4F96938;
  if ( qword_4F96930 )
  {
    ++qword_4F96928;
    qword_4F96930 = 0;
  }
  v3 = dword_4C6F034[0];
  if ( !dword_4C6F034[0] )
    goto LABEL_51;
  if ( pthread_self() == qword_4F96970 )
  {
    if ( v3 != 1 )
      goto LABEL_8;
LABEL_51:
    byte_4F96968 = 0;
    v4 = 0;
    pthread_mutex_unlock(&stru_4F96940);
    return v4;
  }
  if ( qword_4F96970 )
  {
    v18 = 0;
    do
    {
      byte_4F96968 = 0;
      pthread_mutex_unlock(&stru_4F96940);
      if ( v18 > 4 )
      {
        sched_yield();
      }
      else
      {
        for ( LODWORD(cpuset.__bits[0]) = 0; LODWORD(cpuset.__bits[0]) < 1 << v18; ++LODWORD(cpuset.__bits[0]) )
          _mm_pause();
        ++v18;
      }
      if ( pthread_mutex_trylock(&stru_4F96940) )
      {
        sub_130AD90(&unk_4F96900);
        byte_4F96968 = 1;
      }
      ++qword_4F96938;
      if ( qword_4F96930 )
      {
        ++qword_4F96928;
        qword_4F96930 = 0;
      }
    }
    while ( dword_4C6F034[0] );
    goto LABEL_51;
  }
LABEL_8:
  if ( v3 != 2 )
  {
    v4 = sub_1301F00((__int64)v2, a2);
    if ( (_BYTE)v4 )
    {
      byte_4F96968 = 0;
      pthread_mutex_unlock(&stru_4F96940);
      return v4;
    }
  }
  byte_4F96968 = 0;
  pthread_mutex_unlock(&stru_4F96940);
  v5 = sub_1313FC0();
  if ( !v5 )
    return 1;
  dword_4C6F034[0] = 1;
  sched_getaffinity(0, 0x80u, &cpuset);
  v6 = __sched_cpucount(0x80u, &cpuset);
  v7 = &dword_505F9BC;
  if ( v6 == -1 )
    v6 = 1;
  dword_505F9BC = v6;
  if ( unk_4C6F238 != 2 )
  {
    v8 = sysconf(84);
    if ( v8 != sysconf(83) || (sched_getaffinity(0, 0x80u, &cpuset), v8 != __sched_cpucount(0x80u, &cpuset)) )
    {
      if ( !dword_4F96990 )
      {
        unk_4C6F238 = 2;
        sub_130AA40("<jemalloc>: Number of CPUs detected is not deterministic. Per-CPU arena disabled.\n");
        if ( unk_4F969A4 )
          sub_12FC8C0(
            (__int64)"<jemalloc>: Number of CPUs detected is not deterministic. Per-CPU arena disabled.\n",
            (int)&cpuset,
            v21,
            (int)v7,
            v22,
            v23);
        if ( unk_4F969A5 )
LABEL_94:
          abort();
      }
    }
  }
  if ( (unsigned int)sub_2257930(sub_1300C40, sub_1300DD0, sub_1300EB0, v7) )
  {
    sub_130AA40("<jemalloc>: Error in pthread_atfork()\n");
    if ( unk_4F969A5 )
      goto LABEL_94;
    return 1;
  }
  if ( (unsigned __int8)sub_131AE40() )
    return 1;
  if ( pthread_mutex_trylock(&stru_4F96940) )
  {
    sub_130AD90(&unk_4F96900);
    byte_4F96968 = 1;
  }
  ++qword_4F96938;
  if ( v5 != qword_4F96930 )
  {
    ++qword_4F96928;
    qword_4F96930 = v5;
  }
  ++*(_BYTE *)(v5 + 1);
  if ( !*(_BYTE *)(v5 + 816) )
    sub_1313A40(v5);
  if ( unk_4C6F238 == 2 )
  {
    v13 = dword_4F96990;
  }
  else if ( sched_getcpu() < 0 )
  {
    unk_4C6F238 = 2;
    v29 = dword_4F96990;
    if ( !dword_4F96990 )
    {
      v29 = 1;
      if ( dword_505F9BC > 1u )
      {
        v9 = &dword_4C6F0AC;
        v29 = ((dword_4C6F0AC * (dword_505F9BC << 16)) >> 31)
            + ((unsigned int)((dword_4C6F0AC * (unsigned __int64)(unsigned int)(dword_505F9BC << 16)) >> 16) >> 16);
        if ( !v29 )
          v29 = 1;
      }
    }
    sub_130ACF0(
      (unsigned int)"<jemalloc>: perCPU arena getcpu() not available. Setting narenas to %u.\n",
      v29,
      (_DWORD)v9,
      v10,
      v11,
      v12);
    if ( unk_4F969A5 )
      goto LABEL_94;
    v13 = dword_4F96990;
  }
  else
  {
    v20 = dword_505F9BC;
    if ( dword_505F9BC > 0xFFEu )
    {
      sub_130ACF0(
        (unsigned int)"<jemalloc>: narenas w/ percpuarena beyond limit (%d)\n",
        dword_505F9BC,
        (_DWORD)v9,
        v10,
        v11,
        v12);
      if ( unk_4F969A5 )
        goto LABEL_94;
      goto LABEL_70;
    }
    if ( unk_4C6F238 == 1 )
    {
      if ( (dword_505F9BC & 1) == 0 )
        goto LABEL_59;
      sub_130ACF0(
        (unsigned int)"<jemalloc>: invalid configuration -- per physical CPU arena with odd number (%u) of CPUs (no hyper threading?).\n",
        dword_505F9BC,
        (_DWORD)v9,
        v10,
        v11,
        v12);
      if ( unk_4F969A5 )
        goto LABEL_94;
      v20 = dword_505F9BC;
      if ( unk_4C6F238 == 1 )
      {
LABEL_59:
        if ( v20 > 1 )
          v20 = (v20 >> 1) - (((v20 & 1) == 0) - 1);
      }
    }
    v13 = dword_4F96990;
    if ( dword_4F96990 < v20 )
    {
      dword_4F96990 = v20;
      v13 = v20;
    }
  }
  if ( !v13 )
  {
    if ( dword_505F9BC <= 1u )
    {
      dword_4F96990 = 1;
      v13 = 1;
      goto LABEL_32;
    }
    v10 = (dword_4C6F0AC * (unsigned __int64)(unsigned int)(dword_505F9BC << 16)) >> 16;
    v14 = (dword_4C6F0AC * (dword_505F9BC << 16)) >> 31;
    v15 = v14 + WORD1(v10) == 0;
    v13 = v14 + WORD1(v10);
    LODWORD(v9) = 1;
    if ( v15 )
      v13 = 1;
    dword_4F96990 = v13;
  }
  if ( v13 <= 0xFFE )
  {
LABEL_32:
    unk_505F9B8 = v13;
    goto LABEL_33;
  }
  unk_505F9B8 = 4094;
  sub_130ACF0((unsigned int)"<jemalloc>: Reducing narenas to limit (%d)\n", 4094, (_DWORD)v9, v10, v11, v12);
  v13 = unk_505F9B8;
LABEL_33:
  dword_4F96988 = v13;
  if ( (unsigned __int8)sub_1318FF0() )
    _InterlockedAdd(&dword_4F96988, 1u);
  unk_5057900 = sub_1300B70();
  v16 = sub_131BF10();
  if ( (unsigned __int8)sub_131AEA0(v5, v16) )
  {
LABEL_70:
    byte_4F96968 = 0;
    pthread_mutex_unlock(&stru_4F96940);
    v15 = (*(_BYTE *)(v5 + 1))-- == 1;
    if ( v15 )
      sub_1313A40(v5);
    return 1;
  }
  v17 = unk_4C6F238;
  if ( unk_4C6F238 != 2 )
    v17 = unk_4C6F238 + 3;
  unk_4C6F238 = v17;
  v4 = sub_130B0A0();
  if ( (_BYTE)v4 )
  {
    byte_4F96968 = 0;
    pthread_mutex_unlock(&stru_4F96940);
    v15 = (*(_BYTE *)(v5 + 1))-- == 1;
    if ( v15 )
    {
      sub_1313A40(v5);
      return v4;
    }
    return 1;
  }
  dword_4C6F034[0] = 0;
  v24 = 2 * (unk_4F969A1 != 0);
  v25 = 4 * (unk_4F96994 != 0);
  v26 = 8 * (unk_4F96997 != 0);
  v27 = 16 * (unk_4F96996 != 0);
  v15 = (v27 | (unsigned __int8)(v26 | v25 | v24 | byte_4F96978 | unk_4F969A2)) == 0;
  byte_4F96978 |= v27 | v26 | v25 | v24 | unk_4F969A2;
  unk_4C6F030 = !v15;
  v15 = (*(_BYTE *)(v5 + 1))-- == 1;
  if ( v15 )
    sub_1313A40(v5);
  byte_4F96968 = 0;
  pthread_mutex_unlock(&stru_4F96940);
  sub_1314040();
  v28 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v28 = sub_1313D30(v28, 0);
  v4 = byte_4F96B90[0];
  if ( !byte_4F96B90[0] )
    return v4;
  sub_131ADF0(v28);
  return sub_131A460(v28, 0);
}
