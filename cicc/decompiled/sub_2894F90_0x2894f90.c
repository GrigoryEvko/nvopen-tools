// Function: sub_2894F90
// Address: 0x2894f90
//
void __fastcall sub_2894F90(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  char *v11; // rax
  __int64 *v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi

  v6 = *((unsigned int *)a1 + 2);
  v7 = *a1;
  v8 = 5 * v6;
  v9 = (__int64)&(*a1)[22 * v6];
  if ( *a1 != (__int64 *)v9 )
  {
    do
    {
      if ( a2 )
      {
        v11 = (char *)*v7;
        *(_DWORD *)(a2 + 16) = 0;
        *(_DWORD *)(a2 + 20) = 16;
        *(_QWORD *)a2 = v11;
        *(_QWORD *)(a2 + 8) = a2 + 24;
        if ( *((_DWORD *)v7 + 4) )
          sub_2894810(a2 + 8, (char **)v7 + 1, v8, a4, a5, a6);
        *(__m128i *)(a2 + 152) = _mm_loadu_si128((const __m128i *)(v7 + 19));
        *(_BYTE *)(a2 + 168) = *((_BYTE *)v7 + 168);
      }
      v7 += 22;
      a2 += 176;
    }
    while ( (__int64 *)v9 != v7 );
    v12 = *a1;
    v13 = (__int64)&(*a1)[22 * *((unsigned int *)a1 + 2)];
    while ( v12 != (__int64 *)v13 )
    {
      while ( 1 )
      {
        v13 -= 176;
        v14 = *(_QWORD *)(v13 + 8);
        if ( v14 == v13 + 24 )
          break;
        _libc_free(v14);
        if ( v12 == (__int64 *)v13 )
          return;
      }
    }
  }
}
