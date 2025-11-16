// Function: sub_131AEA0
// Address: 0x131aea0
//
__int64 __fastcall sub_131AEA0(__int64 a1, __int64 a2)
{
  signed __int64 v2; // rax
  __int64 v4; // rax
  unsigned int v5; // ebx
  __int64 v6; // rdx
  __int64 v7; // r15
  unsigned __int8 v8; // [rsp+Fh] [rbp-31h]

  v2 = unk_4C6F268;
  if ( unk_4C6F268 > 0xFFFu )
  {
    unk_4C6F268 = 4;
    v2 = 4;
  }
  qword_5260D48[0] = v2;
  byte_5260DD0[0] = byte_4F96B90[0];
  v8 = sub_130AF40((__int64)qword_5260D60);
  if ( !v8 )
  {
    v4 = sub_131C440(a1, a2, 208LL * unk_4C6F268, 64);
    qword_5260DD8 = v4;
    if ( v4 )
    {
      v5 = 0;
      v6 = 0;
      if ( !qword_5260D48[0] )
        return v8;
      while ( 1 )
      {
        v7 = v4 + 208 * v6;
        if ( (unsigned __int8)sub_130AF40(v7 + 56) || pthread_cond_init((pthread_cond_t *)(v7 + 8), 0) )
          break;
        if ( pthread_mutex_trylock((pthread_mutex_t *)(v7 + 120)) )
        {
          sub_130AD90(v7 + 56);
          *(_BYTE *)(v7 + 160) = 1;
        }
        ++*(_QWORD *)(v7 + 112);
        if ( a1 != *(_QWORD *)(v7 + 104) )
        {
          ++*(_QWORD *)(v7 + 96);
          *(_QWORD *)(v7 + 104) = a1;
        }
        *(_DWORD *)(v7 + 168) = 0;
        *(_BYTE *)(v7 + 172) = 0;
        sub_130B0C0((_QWORD *)(v7 + 176), 0);
        *(_QWORD *)(v7 + 184) = 0;
        *(_QWORD *)(v7 + 192) = 0;
        sub_130B140((__int64 *)(v7 + 200), &qword_4287700);
        *(_BYTE *)(v7 + 160) = 0;
        pthread_mutex_unlock((pthread_mutex_t *)(v7 + 120));
        v6 = ++v5;
        if ( (unsigned __int64)v5 >= qword_5260D48[0] )
          return v8;
        v4 = qword_5260DD8;
      }
    }
  }
  return 1;
}
