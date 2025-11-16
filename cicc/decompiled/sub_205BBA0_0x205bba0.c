// Function: sub_205BBA0
// Address: 0x205bba0
//
__int64 __fastcall sub_205BBA0(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // r9d
  __int64 v8; // rbx
  __int64 i; // r12
  __int64 v10; // rax
  int v11; // eax
  __m128i v12; // xmm0
  char v13; // al
  int v14; // eax
  int v15; // r8d
  __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rdi
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // r14
  _QWORD *v26; // r13
  unsigned __int64 v27; // r15
  _QWORD *v28; // r12
  int v30; // [rsp+0h] [rbp-50h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  unsigned __int64 v33; // [rsp+18h] [rbp-38h]

  v1 = (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v2 = (v1 >> 4) | v1;
  v3 = ((v2 >> 8) | v2 | (((v2 >> 8) | v2) >> 16) | (((v2 >> 8) | v2) >> 32)) + 1;
  v4 = 0xFFFFFFFFLL;
  if ( v3 <= 0xFFFFFFFF )
    v4 = v3;
  v30 = v4;
  v31 = malloc(440 * v4);
  if ( !v31 )
    sub_16BD1C0("Allocation failed", 1u);
  v33 = *(_QWORD *)a1 + 440LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v33 )
  {
    v8 = v31;
    for ( i = *(_QWORD *)a1 + 208LL; ; i += 440 )
    {
      if ( v8 )
      {
        *(_DWORD *)v8 = *(_DWORD *)(i - 208);
        *(_DWORD *)(v8 + 4) = *(_DWORD *)(i - 204);
        *(_BYTE *)(v8 + 8) = *(_BYTE *)(i - 200);
        *(_BYTE *)(v8 + 9) = *(_BYTE *)(i - 199);
        *(_BYTE *)(v8 + 10) = *(_BYTE *)(i - 198);
        *(_BYTE *)(v8 + 11) = *(_BYTE *)(i - 197);
        v14 = *(_DWORD *)(i - 196);
        *(_DWORD *)(v8 + 24) = 0;
        *(_DWORD *)(v8 + 12) = v14;
        *(_QWORD *)(v8 + 16) = v8 + 32;
        *(_DWORD *)(v8 + 28) = 1;
        v15 = *(_DWORD *)(i - 184);
        if ( v15 )
          sub_2055660(v8 + 16, i - 192, v5, v6);
        *(_DWORD *)(v8 + 72) = 0;
        *(_QWORD *)(v8 + 64) = v8 + 80;
        *(_DWORD *)(v8 + 76) = 2;
        if ( *(_DWORD *)(i - 136) )
          sub_205B6D0(v8 + 64, i - 144);
        *(_QWORD *)(v8 + 192) = v8 + 208;
        v10 = *(_QWORD *)(i - 16);
        if ( i == v10 )
        {
          *(__m128i *)(v8 + 208) = _mm_loadu_si128((const __m128i *)i);
        }
        else
        {
          *(_QWORD *)(v8 + 192) = v10;
          *(_QWORD *)(v8 + 208) = *(_QWORD *)i;
        }
        *(_QWORD *)(v8 + 200) = *(_QWORD *)(i - 8);
        v11 = *(_DWORD *)(i + 16);
        *(_QWORD *)(i - 16) = i;
        *(_QWORD *)(i - 8) = 0;
        *(_BYTE *)i = 0;
        *(_DWORD *)(v8 + 224) = v11;
        *(_QWORD *)(v8 + 232) = *(_QWORD *)(i + 24);
        *(_BYTE *)(v8 + 240) = *(_BYTE *)(i + 32);
        v12 = _mm_loadu_si128((const __m128i *)(i + 40));
        *(_QWORD *)(v8 + 264) = v8 + 280;
        *(_DWORD *)(v8 + 272) = 0;
        *(_DWORD *)(v8 + 276) = 4;
        *(__m128i *)(v8 + 248) = v12;
        if ( *(_DWORD *)(i + 64) )
          sub_20449C0(v8 + 264, (char **)(i + 56), v5, v6, v15, v7);
        *(_DWORD *)(v8 + 352) = 0;
        *(_QWORD *)(v8 + 344) = v8 + 360;
        *(_DWORD *)(v8 + 356) = 4;
        v6 = *(unsigned int *)(i + 144);
        if ( (_DWORD)v6 )
          sub_2044890(v8 + 344, (char **)(i + 136), v5, v6, v15, v7);
        *(_DWORD *)(v8 + 376) = 0;
        *(_QWORD *)(v8 + 368) = v8 + 384;
        *(_DWORD *)(v8 + 380) = 4;
        v5 = *(unsigned int *)(i + 168);
        if ( (_DWORD)v5 )
          sub_2044C40(v8 + 368, (char **)(i + 160), v5, v6, v15, v7);
        *(_DWORD *)(v8 + 408) = 0;
        *(_QWORD *)(v8 + 400) = v8 + 416;
        *(_DWORD *)(v8 + 412) = 4;
        if ( *(_DWORD *)(i + 200) )
          sub_2044C40(v8 + 400, (char **)(i + 192), v5, v6, v15, v7);
        v13 = *(_BYTE *)(i + 228);
        *(_BYTE *)(v8 + 436) = v13;
        if ( v13 )
          *(_DWORD *)(v8 + 432) = *(_DWORD *)(i + 224);
      }
      v8 += 440;
      if ( v33 == i + 232 )
        break;
    }
    v33 = *(_QWORD *)a1;
    v16 = *(_QWORD *)a1 + 440LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v16 )
    {
      do
      {
        v16 -= 440;
        v17 = *(_QWORD *)(v16 + 400);
        if ( v17 != v16 + 416 )
          _libc_free(v17);
        v18 = *(_QWORD *)(v16 + 368);
        if ( v18 != v16 + 384 )
          _libc_free(v18);
        v19 = *(_QWORD *)(v16 + 344);
        if ( v19 != v16 + 360 )
          _libc_free(v19);
        v20 = *(_QWORD *)(v16 + 264);
        if ( v20 != v16 + 280 )
          _libc_free(v20);
        v21 = *(_QWORD *)(v16 + 192);
        if ( v21 != v16 + 208 )
          j_j___libc_free_0(v21, *(_QWORD *)(v16 + 208) + 1LL);
        v22 = *(_QWORD *)(v16 + 64);
        v23 = v22 + 56LL * *(unsigned int *)(v16 + 72);
        if ( v22 != v23 )
        {
          do
          {
            v24 = *(unsigned int *)(v23 - 40);
            v25 = *(_QWORD *)(v23 - 48);
            v23 -= 56LL;
            v24 *= 32;
            v26 = (_QWORD *)(v25 + v24);
            if ( v25 != v25 + v24 )
            {
              do
              {
                v26 -= 4;
                if ( (_QWORD *)*v26 != v26 + 2 )
                  j_j___libc_free_0(*v26, v26[2] + 1LL);
              }
              while ( (_QWORD *)v25 != v26 );
              v25 = *(_QWORD *)(v23 + 8);
            }
            if ( v25 != v23 + 24 )
              _libc_free(v25);
          }
          while ( v22 != v23 );
          v22 = *(_QWORD *)(v16 + 64);
        }
        if ( v22 != v16 + 80 )
          _libc_free(v22);
        v27 = *(_QWORD *)(v16 + 16);
        v28 = (_QWORD *)(v27 + 32LL * *(unsigned int *)(v16 + 24));
        if ( (_QWORD *)v27 != v28 )
        {
          do
          {
            v28 -= 4;
            if ( (_QWORD *)*v28 != v28 + 2 )
              j_j___libc_free_0(*v28, v28[2] + 1LL);
          }
          while ( (_QWORD *)v27 != v28 );
          v27 = *(_QWORD *)(v16 + 16);
        }
        if ( v27 != v16 + 32 )
          _libc_free(v27);
      }
      while ( v16 != v33 );
      v33 = *(_QWORD *)a1;
    }
  }
  if ( v33 != a1 + 16 )
    _libc_free(v33);
  *(_QWORD *)a1 = v31;
  *(_DWORD *)(a1 + 12) = v30;
  return a1;
}
