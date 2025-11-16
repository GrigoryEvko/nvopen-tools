// Function: sub_130EE60
// Address: 0x130ee60
//
int __fastcall sub_130EE60(__int64 a1, __int64 a2, __int64 a3)
{
  int result; // eax
  unsigned __int64 v5; // r12
  __int64 v6; // r15
  unsigned int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // r15
  _DWORD *v11; // [rsp+8h] [rbp-48h]
  __int64 *v12; // [rsp+18h] [rbp-38h]

  v12 = (__int64 *)(a3 + 8);
  result = a3 + 36;
  v11 = (_DWORD *)(a3 + 36);
  if ( *(_QWORD *)(a2 + 64) )
  {
    v5 = 0;
    do
    {
      v9 = 144 * v5;
      v10 = 144 * v5 + *(_QWORD *)(a2 + 104);
      if ( pthread_mutex_trylock((pthread_mutex_t *)(v10 + 64)) )
      {
        sub_130AD90(v10);
        *(_BYTE *)(v10 + 104) = 1;
      }
      ++*(_QWORD *)(v10 + 56);
      if ( a1 != *(_QWORD *)(v10 + 48) )
      {
        ++*(_QWORD *)(v10 + 40);
        *(_QWORD *)(v10 + 48) = a1;
      }
      v6 = v9 + *(_QWORD *)(a2 + 104);
      sub_130B1D0((_QWORD *)a3, (__int64 *)v6);
      if ( (int)sub_130B150((_QWORD *)(v6 + 8), v12) > 0 )
        sub_130B140(v12, (__int64 *)(v6 + 8));
      *(_QWORD *)(a3 + 16) += *(_QWORD *)(v6 + 16);
      *(_QWORD *)(a3 + 24) += *(_QWORD *)(v6 + 24);
      v7 = *(_DWORD *)(v6 + 32);
      if ( *(_DWORD *)(a3 + 32) < v7 )
        *(_DWORD *)(a3 + 32) = v7;
      ++v5;
      *v11 = 0;
      *(_QWORD *)(a3 + 40) += *(_QWORD *)(v6 + 40);
      *(_QWORD *)(a3 + 56) += *(_QWORD *)(v6 + 56);
      v8 = *(_QWORD *)(a2 + 104) + v9;
      *(_BYTE *)(v8 + 104) = 0;
      result = pthread_mutex_unlock((pthread_mutex_t *)(v8 + 64));
    }
    while ( *(_QWORD *)(a2 + 64) > v5 );
  }
  return result;
}
