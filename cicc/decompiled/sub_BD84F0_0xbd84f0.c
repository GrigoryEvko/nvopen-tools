// Function: sub_BD84F0
// Address: 0xbd84f0
//
__int64 __fastcall sub_BD84F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rbx
  _QWORD *v7; // rdi

  v2 = *a1;
  if ( !*((_DWORD *)a1 + 3) )
    return _libc_free(*a1, a2);
  v4 = *((unsigned int *)a1 + 2);
  if ( (_DWORD)v4 )
  {
    v5 = 8 * v4;
    v6 = 0;
    do
    {
      v7 = *(_QWORD **)(v2 + v6);
      if ( v7 != (_QWORD *)-8LL )
      {
        if ( v7 )
        {
          a2 = *v7 + 17LL;
          sub_C7D6A0(v7, a2, 8);
          v2 = *a1;
        }
      }
      v6 += 8;
    }
    while ( v5 != v6 );
  }
  return _libc_free(v2, a2);
}
