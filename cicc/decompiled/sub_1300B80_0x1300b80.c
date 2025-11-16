// Function: sub_1300B80
// Address: 0x1300b80
//
__int64 __fastcall sub_1300B80(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r14
  int v7; // edx
  int v8; // ecx
  int v9; // r8d
  int v10; // r9d

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
  v4 = sub_1301030(a1, a2, a3);
  stru_5057960[1].__size[0] = 0;
  v5 = v4;
  pthread_mutex_unlock(stru_5057960);
  if ( a2 && !(unsigned __int8)sub_1319070(a2) && (unsigned __int8)sub_131A460(a1, a2) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: error in background thread creation for arena %u. Abort.\n",
      a2,
      v7,
      v8,
      v9,
      v10);
    abort();
  }
  return v5;
}
