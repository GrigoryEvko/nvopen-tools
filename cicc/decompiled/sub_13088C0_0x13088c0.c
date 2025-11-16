// Function: sub_13088C0
// Address: 0x13088c0
//
__int64 sub_13088C0()
{
  unsigned __int64 v0; // r13
  int v1; // edx
  __int64 *v2; // rax
  __int64 *v3; // r12
  __int64 v4; // r14
  _QWORD *v5; // rbx
  __int64 *v7; // [rsp+8h] [rbp-38h]

  if ( unk_4F96B58 )
  {
    v0 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      v0 = sub_1313D30(v0, 0);
  }
  else
  {
    v0 = 0;
  }
  v1 = sub_1300B70();
  if ( v1 )
  {
    v2 = qword_50579C0;
    v3 = &qword_50579C0[1];
    v7 = &qword_50579C0[(unsigned int)(v1 - 1) + 1];
    while ( 1 )
    {
      v4 = *v2;
      if ( *v2 )
      {
        if ( pthread_mutex_trylock((pthread_mutex_t *)(v4 + 10472)) )
        {
          sub_130AD90(v4 + 10408);
          *(_BYTE *)(v4 + 10512) = 1;
        }
        ++*(_QWORD *)(v4 + 10464);
        if ( v0 != *(_QWORD *)(v4 + 10456) )
        {
          ++*(_QWORD *)(v4 + 10448);
          *(_QWORD *)(v4 + 10456) = v0;
        }
        v5 = *(_QWORD **)(v4 + 10392);
        do
        {
          if ( !v5 )
            break;
          sub_1311650(v0, v5[21], v4);
          v5 = (_QWORD *)*v5;
        }
        while ( v5 != *(_QWORD **)(v4 + 10392) );
        *(_BYTE *)(v4 + 10512) = 0;
        pthread_mutex_unlock((pthread_mutex_t *)(v4 + 10472));
        v2 = v3;
        if ( v7 == v3 )
          return sub_1308810(0, 0, (__int64)byte_4F96A20);
      }
      else
      {
        v2 = v3;
        if ( v7 == v3 )
          return sub_1308810(0, 0, (__int64)byte_4F96A20);
      }
      ++v3;
    }
  }
  return sub_1308810(0, 0, (__int64)byte_4F96A20);
}
