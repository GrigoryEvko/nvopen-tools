// Function: sub_16CC1C0
// Address: 0x16cc1c0
//
bool __fastcall sub_16CC1C0(
        pthread_mutex_t **a1,
        struct sigaction *p_act,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  pthread_mutex_t **v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // ebx
  char *i; // r15
  __int64 v13; // r12
  struct sigaction *v14; // r12
  unsigned int v15; // r12d
  unsigned int *j; // r15
  __int64 v17; // rbx
  struct sigaction *v18; // rbx
  __int64 v20; // [rsp+8h] [rbp-D8h]
  struct sigaction act; // [rsp+10h] [rbp-D0h] BYREF

  if ( !qword_4FA0650 )
  {
    p_act = (struct sigaction *)sub_160CFB0;
    a1 = (pthread_mutex_t **)&qword_4FA0650;
    sub_16C1EA0((__int64)&qword_4FA0650, sub_160CFB0, (__int64)sub_160D0B0, a4, a5, a6);
  }
  v6 = (pthread_mutex_t **)qword_4FA0650;
  v20 = qword_4FA0650;
  if ( (unsigned __int8)sub_16D5D40(a1, p_act, a3, a4, a5, a6) )
  {
    a1 = v6;
    sub_16C30C0(v6);
  }
  else
  {
    ++*(_DWORD *)(v20 + 8);
  }
  if ( !dword_4FA1080 )
  {
    v11 = 1;
    for ( i = (char *)&unk_42AF130; ; v11 = *(_DWORD *)i )
    {
      v13 = 160LL * (unsigned int)dword_4FA1080;
      act.sa_handler = (__sighandler_t)sub_16CC630;
      act.sa_flags = -939524096;
      sigemptyset(&act.sa_mask);
      v14 = (struct sigaction *)((char *)&stru_4FA0680 + v13);
      sigaction(v11, &act, v14);
      LODWORD(v14[1].sa_handler) = v11;
      _InterlockedAdd(&dword_4FA1080, 1u);
      i += 4;
      if ( i == "No available targets are compatible with this triple." )
        break;
    }
    v15 = 4;
    for ( j = (unsigned int *)&unk_42AF100; ; v15 = *j )
    {
      v17 = 160LL * (unsigned int)dword_4FA1080;
      act.sa_handler = (__sighandler_t)sub_16CC630;
      act.sa_flags = -939524096;
      sigemptyset(&act.sa_mask);
      p_act = &act;
      a1 = (pthread_mutex_t **)v15;
      v18 = (struct sigaction *)((char *)&stru_4FA0680 + v17);
      sigaction(v15, &act, v18);
      LODWORD(v18[1].sa_handler) = v15;
      _InterlockedAdd(&dword_4FA1080, 1u);
      if ( ++j == (unsigned int *)&unk_42AF128 )
        break;
    }
  }
  if ( (unsigned __int8)sub_16D5D40(a1, p_act, v7, v8, v9, v10) )
    return sub_16C30E0((pthread_mutex_t **)v20);
  --*(_DWORD *)(v20 + 8);
  return v20;
}
