// Function: sub_337BFD0
// Address: 0x337bfd0
//
void __fastcall sub_337BFD0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 i; // r12
  __int64 v9; // rax
  int v10; // eax
  __m128i v11; // xmm0
  __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // r12
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // r15
  __int64 v22; // rbx
  unsigned __int64 v23; // r14
  unsigned __int64 *v24; // rbx
  unsigned __int64 v25; // r13
  unsigned __int64 *v26; // rbx
  __int64 v27; // [rsp+8h] [rbp-38h]

  v6 = *a1 + 448LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v6 )
  {
    for ( i = *a1 + 208; ; i += 448 )
    {
      if ( a2 )
      {
        *(_DWORD *)a2 = *(_DWORD *)(i - 208);
        *(_DWORD *)(a2 + 4) = *(_DWORD *)(i - 204);
        *(_BYTE *)(a2 + 8) = *(_BYTE *)(i - 200);
        *(_BYTE *)(a2 + 9) = *(_BYTE *)(i - 199);
        *(_BYTE *)(a2 + 10) = *(_BYTE *)(i - 198);
        *(_BYTE *)(a2 + 11) = *(_BYTE *)(i - 197);
        v13 = *(_DWORD *)(i - 196);
        *(_DWORD *)(a2 + 24) = 0;
        *(_DWORD *)(a2 + 12) = v13;
        *(_QWORD *)(a2 + 16) = a2 + 32;
        *(_DWORD *)(a2 + 28) = 1;
        if ( *(_DWORD *)(i - 184) )
          sub_337B320(a2 + 16, i - 192);
        *(_DWORD *)(a2 + 72) = 0;
        *(_QWORD *)(a2 + 64) = a2 + 80;
        *(_DWORD *)(a2 + 76) = 2;
        if ( *(_DWORD *)(i - 136) )
          sub_337BAD0(a2 + 64, i - 144);
        *(_QWORD *)(a2 + 192) = a2 + 208;
        v9 = *(_QWORD *)(i - 16);
        if ( v9 == i )
        {
          *(__m128i *)(a2 + 208) = _mm_loadu_si128((const __m128i *)i);
        }
        else
        {
          *(_QWORD *)(a2 + 192) = v9;
          *(_QWORD *)(a2 + 208) = *(_QWORD *)i;
        }
        *(_QWORD *)(a2 + 200) = *(_QWORD *)(i - 8);
        v10 = *(_DWORD *)(i + 16);
        *(_QWORD *)(i - 16) = i;
        *(_QWORD *)(i - 8) = 0;
        *(_BYTE *)i = 0;
        *(_DWORD *)(a2 + 224) = v10;
        *(_QWORD *)(a2 + 232) = *(_QWORD *)(i + 24);
        *(_WORD *)(a2 + 240) = *(_WORD *)(i + 32);
        v11 = _mm_loadu_si128((const __m128i *)(i + 40));
        *(_QWORD *)(a2 + 264) = a2 + 280;
        *(_DWORD *)(a2 + 272) = 0;
        *(_DWORD *)(a2 + 276) = 4;
        *(__m128i *)(a2 + 248) = v11;
        v12 = *(unsigned int *)(i + 64);
        if ( (_DWORD)v12 )
          sub_3365840(a2 + 264, (char **)(i + 56), a3, v12, a5, a6);
        *(_QWORD *)(a2 + 352) = 0;
        *(_QWORD *)(a2 + 344) = a2 + 368;
        *(_QWORD *)(a2 + 360) = 4;
        if ( *(_QWORD *)(i + 144) )
          sub_33656C0(a2 + 344, (char **)(i + 136), a3, v12, a5, a6);
        *(_DWORD *)(a2 + 384) = 0;
        *(_QWORD *)(a2 + 376) = a2 + 392;
        *(_DWORD *)(a2 + 388) = 4;
        a3 = *(unsigned int *)(i + 176);
        if ( (_DWORD)a3 )
          sub_3365560(a2 + 376, (char **)(i + 168), a3, v12, a5, a6);
        *(_DWORD *)(a2 + 416) = 0;
        *(_QWORD *)(a2 + 408) = a2 + 424;
        *(_DWORD *)(a2 + 420) = 4;
        if ( *(_DWORD *)(i + 208) )
          sub_33659A0(a2 + 408, (char **)(i + 200), a3, v12, a5, a6);
        *(_QWORD *)(a2 + 440) = *(_QWORD *)(i + 232);
      }
      a2 += 448;
      if ( v6 == i + 240 )
        break;
    }
    v27 = *a1;
    v14 = *a1 + 448LL * *((unsigned int *)a1 + 2);
    if ( v14 != *a1 )
    {
      do
      {
        v14 -= 448;
        v15 = *(_QWORD *)(v14 + 408);
        if ( v15 != v14 + 424 )
          _libc_free(v15);
        v16 = *(_QWORD *)(v14 + 376);
        if ( v16 != v14 + 392 )
          _libc_free(v16);
        v17 = *(_QWORD *)(v14 + 344);
        if ( v17 != v14 + 368 )
          _libc_free(v17);
        v18 = *(_QWORD *)(v14 + 264);
        if ( v18 != v14 + 280 )
          _libc_free(v18);
        v19 = *(_QWORD *)(v14 + 192);
        if ( v19 != v14 + 208 )
          j_j___libc_free_0(v19);
        v20 = *(_QWORD *)(v14 + 64);
        v21 = v20 + 56LL * *(unsigned int *)(v14 + 72);
        if ( v20 != v21 )
        {
          do
          {
            v22 = *(unsigned int *)(v21 - 40);
            v23 = *(_QWORD *)(v21 - 48);
            v21 -= 56LL;
            v24 = (unsigned __int64 *)(v23 + 32 * v22);
            if ( (unsigned __int64 *)v23 != v24 )
            {
              do
              {
                v24 -= 4;
                if ( (unsigned __int64 *)*v24 != v24 + 2 )
                  j_j___libc_free_0(*v24);
              }
              while ( (unsigned __int64 *)v23 != v24 );
              v23 = *(_QWORD *)(v21 + 8);
            }
            if ( v23 != v21 + 24 )
              _libc_free(v23);
          }
          while ( v20 != v21 );
          v20 = *(_QWORD *)(v14 + 64);
        }
        if ( v20 != v14 + 80 )
          _libc_free(v20);
        v25 = *(_QWORD *)(v14 + 16);
        v26 = (unsigned __int64 *)(v25 + 32LL * *(unsigned int *)(v14 + 24));
        if ( (unsigned __int64 *)v25 != v26 )
        {
          do
          {
            v26 -= 4;
            if ( (unsigned __int64 *)*v26 != v26 + 2 )
              j_j___libc_free_0(*v26);
          }
          while ( (unsigned __int64 *)v25 != v26 );
          v25 = *(_QWORD *)(v14 + 16);
        }
        if ( v25 != v14 + 32 )
          _libc_free(v25);
      }
      while ( v14 != v27 );
    }
  }
}
