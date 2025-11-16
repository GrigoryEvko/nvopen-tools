// Function: sub_1302AE0
// Address: 0x1302ae0
//
__int64 __fastcall sub_1302AE0(__int64 a1, char a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // r15d
  __int64 v4; // r14
  _QWORD *v5; // rbx
  int v6; // r8d
  __int64 v7; // rbx
  __int64 v8; // r14
  int v9; // eax
  int v10; // r8d
  __int64 v11; // rbx
  unsigned int v12; // r13d
  int v13; // edx
  int v14; // ecx
  int v15; // r8d
  int v16; // r9d
  __int64 v18; // rax
  unsigned int v19; // eax
  int v20; // [rsp+10h] [rbp-50h]
  unsigned int v21; // [rsp+18h] [rbp-48h]
  __int64 *v22; // [rsp+18h] [rbp-48h]
  __int16 v23; // [rsp+26h] [rbp-3Ah]
  _QWORD v24[7]; // [rsp+28h] [rbp-38h]

  if ( unk_4C6F238 > 2u )
  {
    v19 = sched_getcpu();
    if ( unk_4C6F238 != 3 && v19 >= dword_505F9BC >> 1 )
      v19 -= dword_505F9BC >> 1;
    v8 = qword_50579C0[v19];
    if ( !v8 )
      v8 = sub_1300B80(a1, v19, (__int64)&off_49E8000);
    sub_1300F90(a1, *(_DWORD *)(v8 + 78928), 0);
    sub_1300F90(a1, *(_DWORD *)(v8 + 78928), 1u);
  }
  else
  {
    v2 = unk_505F9B8;
    if ( unk_505F9B8 <= 1u )
    {
      v8 = qword_50579C0[0];
      sub_1300F90(a1, 0, 0);
      sub_1300F90(a1, 0, 1u);
    }
    else
    {
      v24[0] = 0;
      v23 = 0;
      if ( pthread_mutex_trylock(stru_5057960) )
      {
        sub_130AD90(&unk_5057920);
        stru_5057960[1].__size[0] = 1;
      }
      ++unk_5057958;
      if ( a1 != unk_5057950 )
      {
        ++unk_5057948;
        unk_5057950 = a1;
      }
      if ( unk_505F9B8 <= 1u )
      {
        v6 = 0;
      }
      else
      {
        v3 = 1;
        do
        {
          while ( 1 )
          {
            v4 = 0;
            v5 = &qword_50579C0[v3];
            if ( !*v5 )
              break;
            while ( 1 )
            {
              v21 = sub_1317080(*v5, (unsigned int)v4);
              if ( v21 < (unsigned int)sub_1317080(qword_50579C0[*((unsigned int *)v24 + v4)], (unsigned int)v4) )
                *((_DWORD *)v24 + v4) = v3;
              if ( v4 == 1 )
                break;
              v4 = 1;
            }
            if ( unk_505F9B8 <= ++v3 )
              goto LABEL_17;
          }
          if ( unk_505F9B8 == v2 )
            v2 = v3;
          ++v3;
        }
        while ( unk_505F9B8 > v3 );
LABEL_17:
        v6 = v24[0];
      }
      v7 = 0;
      v8 = 0;
      while ( 1 )
      {
        v20 = v6;
        v22 = &qword_50579C0[v6];
        v9 = sub_1317080(*v22, (unsigned int)v7);
        v10 = v20;
        if ( v9 && unk_505F9B8 != v2 )
        {
          *((_DWORD *)v24 + v7) = v2;
          v18 = sub_1301030(a1, v2, (__int64)&off_49E8000);
          if ( !v18 )
          {
            v8 = 0;
            stru_5057960[1].__size[0] = 0;
            pthread_mutex_unlock(stru_5057960);
            return v8;
          }
          *((_BYTE *)&v23 + v7) = 1;
          v10 = v2;
          if ( a2 == (_BYTE)v7 )
            v8 = v18;
        }
        else if ( a2 == (_BYTE)v7 )
        {
          v8 = *v22;
        }
        sub_1300F90(a1, v10, v7);
        if ( v7 == 1 )
          break;
        v6 = HIDWORD(v24[0]);
        v7 = 1;
      }
      v11 = 0;
      stru_5057960[1].__size[0] = 0;
      pthread_mutex_unlock(stru_5057960);
      if ( (_BYTE)v23 )
        goto LABEL_29;
      while ( v11 != 1 )
      {
        v11 = 1;
        if ( HIBYTE(v23) )
        {
LABEL_29:
          v12 = *((_DWORD *)v24 + v11);
          if ( v12 && !(unsigned __int8)sub_1319070(v12) )
          {
            if ( (unsigned __int8)sub_131A460(a1, v12) )
            {
              sub_130ACF0(
                (unsigned int)"<jemalloc>: error in background thread creation for arena %u. Abort.\n",
                v12,
                v13,
                v14,
                v15,
                v16);
              abort();
            }
          }
        }
      }
    }
  }
  return v8;
}
