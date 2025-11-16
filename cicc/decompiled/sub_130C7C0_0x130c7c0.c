// Function: sub_130C7C0
// Address: 0x130c7c0
//
__int64 __fastcall sub_130C7C0(__int64 a1, __int64 a2, int a3, __int64 a4, int a5)
{
  __int64 v7; // r15
  char v8; // r8
  __int64 result; // rax
  __int64 v10; // [rsp+20h] [rbp-50h]
  volatile signed __int64 *v11; // [rsp+28h] [rbp-48h]
  __int64 v12[7]; // [rsp+38h] [rbp-38h] BYREF

  if ( a3 == 1 )
  {
    v7 = a2 + 58648;
    v11 = *(volatile signed __int64 **)(a2 + 62224);
    v10 = a2 + 56;
  }
  else
  {
    v7 = a2 + 60432;
    v11 = (volatile signed __int64 *)(*(_QWORD *)(a2 + 62224) + 24LL);
    v10 = a2 + 19496;
  }
  v8 = sub_133D960(a4);
  result = 1;
  if ( v8 )
  {
    if ( pthread_mutex_trylock((pthread_mutex_t *)(v7 + 64)) )
    {
      sub_130AD90(v7);
      *(_BYTE *)(v7 + 104) = 1;
    }
    ++*(_QWORD *)(v7 + 56);
    if ( a1 != *(_QWORD *)(v7 + 48) )
    {
      ++*(_QWORD *)(v7 + 40);
      *(_QWORD *)(v7 + 48) = a1;
    }
    sub_130B270(v12);
    sub_133D860(v7, v12, a4);
    sub_130C6A0(a1, a2, v7, v11, v10, a5);
    *(_BYTE *)(v7 + 104) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(v7 + 64));
    return 0;
  }
  return result;
}
