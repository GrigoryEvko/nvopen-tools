// Function: sub_1314970
// Address: 0x1314970
//
int __fastcall sub_1314970(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // rbx
  int result; // eax
  int v8; // eax
  pthread_mutex_t *v9; // r8
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // rcx
  pthread_mutex_t *v13; // r8
  __int64 v14; // rax
  _QWORD v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = unk_5260DD8 + 208 * ((unsigned __int64)*(unsigned int *)(a2 + 78928) % unk_5260D48);
  result = pthread_mutex_trylock((pthread_mutex_t *)(v6 + 120));
  if ( result )
  {
    *(_BYTE *)(v6 + 160) = 1;
  }
  else
  {
    ++*(_QWORD *)(v6 + 112);
    if ( a1 != *(_QWORD *)(v6 + 104) )
    {
      ++*(_QWORD *)(v6 + 96);
      *(_QWORD *)(v6 + 104) = a1;
    }
    if ( (unsigned __int8)sub_131A7B0(v6) )
    {
      if ( *(_BYTE *)(v6 + 172) )
      {
        sub_131A7C0(v6, 0);
      }
      else
      {
        v8 = pthread_mutex_trylock((pthread_mutex_t *)(a3 + 64));
        v9 = (pthread_mutex_t *)(a3 + 64);
        if ( v8 )
        {
          *(_BYTE *)(a3 + 104) = 1;
        }
        else
        {
          ++*(_QWORD *)(a3 + 56);
          if ( a1 != *(_QWORD *)(a3 + 48) )
          {
            ++*(_QWORD *)(a3 + 40);
            *(_QWORD *)(a3 + 48) = a1;
          }
          if ( *(__int64 *)(a3 + 120) <= 0
            || (v10 = sub_130B0E0(v6 + 176),
                sub_130B0C0(v15, v10),
                v11 = sub_130B150(v15, (_QWORD *)(a3 + 136)),
                v9 = (pthread_mutex_t *)(a3 + 64),
                v11 <= 0) )
          {
            *(_BYTE *)(a3 + 104) = 0;
            pthread_mutex_unlock(v9);
          }
          else
          {
            sub_130B1F0(v15, (__int64 *)(a3 + 136));
            v13 = (pthread_mutex_t *)(a3 + 64);
            if ( a4 )
            {
              v14 = sub_133D980(a3, v15, a4, v12, a3 + 64);
              v13 = (pthread_mutex_t *)(a3 + 64);
              *(_QWORD *)(v6 + 184) += v14;
            }
            *(_BYTE *)(a3 + 104) = 0;
            pthread_mutex_unlock(v13);
            if ( *(_QWORD *)(v6 + 184) > 0x400u )
            {
              *(_QWORD *)(v6 + 184) = 0;
              sub_131A7C0(v6, v15);
            }
          }
        }
      }
    }
    *(_BYTE *)(v6 + 160) = 0;
    return pthread_mutex_unlock((pthread_mutex_t *)(v6 + 120));
  }
  return result;
}
