// Function: sub_37BF480
// Address: 0x37bf480
//
void __fastcall sub_37BF480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned __int64 *v11; // r12
  unsigned __int64 *v12; // rbx

  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(unsigned __int64 **)a1;
  v8 = 5 * v6;
  v9 = *(_QWORD *)a1 + 88 * v6;
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      if ( a2 )
      {
        *(_DWORD *)(a2 + 8) = 0;
        *(_QWORD *)a2 = a2 + 16;
        *(_DWORD *)(a2 + 12) = 1;
        if ( *((_DWORD *)v7 + 2) )
          sub_37B6100(a2, (char **)v7, v8, a4, a5, a6);
        *(_DWORD *)(a2 + 64) = *((_DWORD *)v7 + 16);
        *(__m128i *)(a2 + 72) = _mm_loadu_si128((const __m128i *)(v7 + 9));
      }
      v7 += 11;
      a2 += 88;
    }
    while ( (unsigned __int64 *)v9 != v7 );
    v11 = *(unsigned __int64 **)a1;
    v12 = (unsigned __int64 *)(*(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v12 )
    {
      do
      {
        v12 -= 11;
        if ( (unsigned __int64 *)*v12 != v12 + 2 )
          _libc_free(*v12);
      }
      while ( v12 != v11 );
    }
  }
}
