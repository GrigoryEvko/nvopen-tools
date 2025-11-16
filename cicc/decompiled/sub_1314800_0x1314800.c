// Function: sub_1314800
// Address: 0x1314800
//
int __fastcall sub_1314800(__int64 a1, int *a2)
{
  void *v2; // rax
  int v4; // eax
  int v5; // r8d
  _QWORD *v6; // rax
  int i; // ebx
  __int64 v8; // r12
  __int64 *v9; // r12
  __int64 *v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rbx
  _QWORD *v14; // [rsp-158h] [rbp-158h]
  int v15; // [rsp-150h] [rbp-150h]
  int v16; // [rsp-14Ch] [rbp-14Ch]
  int v17; // [rsp-13Ch] [rbp-13Ch] BYREF
  __int64 v18[39]; // [rsp-138h] [rbp-138h] BYREF

  v2 = &unk_4C6F2C8;
  if ( !unk_4C6F2C8 )
  {
    v4 = *a2;
    v17 = 0;
    v16 = v4;
    v5 = sub_1300B70();
    if ( v5 )
    {
      v6 = qword_50579C0;
      for ( i = 0; i != v5; ++i )
      {
        if ( v16 != i )
        {
          v8 = *v6;
          if ( *v6 )
          {
            v14 = v6;
            v15 = v5;
            sub_1314710(a1, v8 + 10728, v18, &v17);
            sub_1314710(a1, v8 + 30168, v18, &v17);
            sub_1314710(a1, v8 + 49608, v18, &v17);
            v6 = v14;
            v5 = v15;
          }
        }
        ++v6;
      }
    }
    LODWORD(v2) = v17;
    if ( v17 )
    {
      v9 = v18;
      v10 = &v18[(unsigned int)(v17 - 1) + 1];
      do
      {
        v12 = *v9;
        if ( pthread_mutex_trylock((pthread_mutex_t *)(*v9 + 64)) )
        {
          sub_130AD90(v12);
          *(_BYTE *)(v12 + 104) = 1;
        }
        ++*(_QWORD *)(v12 + 56);
        if ( a1 != *(_QWORD *)(v12 + 48) )
        {
          ++*(_QWORD *)(v12 + 40);
          *(_QWORD *)(v12 + 48) = a1;
        }
        v11 = *v9++;
        *(_BYTE *)(v11 + 104) = 0;
        LODWORD(v2) = pthread_mutex_unlock((pthread_mutex_t *)(v11 + 64));
      }
      while ( v10 != v9 );
    }
  }
  return (int)v2;
}
