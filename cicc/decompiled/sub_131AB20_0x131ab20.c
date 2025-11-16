// Function: sub_131AB20
// Address: 0x131ab20
//
__int64 __fastcall sub_131AB20(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // ebx
  __int64 v4; // rax
  pthread_mutex_t *v5; // r15
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 *v13; // [rsp+8h] [rbp-58h]
  __int64 *v14; // [rsp+10h] [rbp-50h]
  __int64 *v15; // [rsp+18h] [rbp-48h]
  unsigned __int64 v16; // [rsp+20h] [rbp-40h]

  if ( pthread_mutex_trylock(&stru_5260DA0) )
  {
    sub_130AD90((__int64)qword_5260D60);
    unk_5260DC8 = 1;
  }
  ++unk_5260D98;
  if ( a1 != unk_5260D90 )
  {
    ++unk_5260D88;
    unk_5260D90 = a1;
  }
  if ( byte_5260DD0[0] )
  {
    v13 = (unsigned __int64 *)(a2 + 16);
    sub_130B140((__int64 *)(a2 + 16), &qword_4287700);
    v15 = (__int64 *)(a2 + 24);
    *(_OWORD *)(a2 + 24) = 0;
    v2 = unk_5260D40;
    *(_OWORD *)(a2 + 40) = 0;
    *(_OWORD *)(a2 + 56) = 0;
    *(_QWORD *)a2 = v2;
    *(_OWORD *)(a2 + 72) = 0;
    if ( qword_5260D48[0] )
    {
      v3 = 0;
      v4 = 0;
      v16 = 0;
      v14 = (__int64 *)(a2 + 32);
      do
      {
        v5 = (pthread_mutex_t *)(qword_5260DD8 + 208 * v4 + 120);
        v6 = qword_5260DD8 + 208 * v4;
        if ( pthread_mutex_trylock(v5) )
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
          if ( *(_DWORD *)(v6 + 168) )
          {
            v16 += *(_QWORD *)(v6 + 192);
            sub_130B1D0(v13, (__int64 *)(v6 + 200));
            if ( (int)sub_130B150((_QWORD *)(v6 + 56), v15) > 0 )
              sub_130B140(v15, (__int64 *)(v6 + 56));
            if ( (int)sub_130B150((_QWORD *)(v6 + 64), v14) > 0 )
              sub_130B140(v14, (__int64 *)(v6 + 64));
            v7 = *(_QWORD *)(v6 + 72);
            if ( v7 > *(_QWORD *)(a2 + 40) )
              *(_QWORD *)(a2 + 40) = v7;
            v8 = *(_QWORD *)(v6 + 80);
            if ( v8 > *(_QWORD *)(a2 + 48) )
              *(_QWORD *)(a2 + 48) = v8;
            v9 = *(_DWORD *)(v6 + 88);
            if ( v9 > *(_DWORD *)(a2 + 56) )
              *(_DWORD *)(a2 + 56) = v9;
            v10 = *(_QWORD *)(v6 + 96);
            if ( v10 > *(_QWORD *)(a2 + 64) )
              *(_QWORD *)(a2 + 64) = v10;
            v11 = *(_QWORD *)(v6 + 112);
            if ( v11 > *(_QWORD *)(a2 + 80) )
              *(_QWORD *)(a2 + 80) = v11;
          }
          *(_BYTE *)(v6 + 160) = 0;
          pthread_mutex_unlock(v5);
        }
        v4 = ++v3;
      }
      while ( (unsigned __int64)v3 < qword_5260D48[0] );
      *(_QWORD *)(a2 + 8) = v16;
      if ( v16 )
        sub_130B210(v13, v16);
    }
    else
    {
      *(_QWORD *)(a2 + 8) = 0;
    }
    unk_5260DC8 = 0;
    pthread_mutex_unlock(&stru_5260DA0);
    return 0;
  }
  else
  {
    unk_5260DC8 = 0;
    pthread_mutex_unlock(&stru_5260DA0);
    return 1;
  }
}
