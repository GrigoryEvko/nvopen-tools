// Function: sub_3012CF0
// Address: 0x3012cf0
//
void __fastcall sub_3012CF0(__int64 a1, _BYTE *a2, int a3, int a4, const __m128i *a5, unsigned __int64 a6)
{
  const __m128i *v6; // r13
  __int64 *v7; // rbx
  __int64 v8; // rcx
  _QWORD *v9; // rax
  unsigned __int8 *v10; // rax
  __int64 v11; // rax
  int *v12; // rdx
  __m128i *v13; // rax
  __int64 v14; // r15
  unsigned __int8 *v15; // r14
  const void **v16; // r12
  __int64 v17; // rbx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // rbx
  int v23; // eax
  const void *v24; // rdx
  void *v25; // rdi
  unsigned int v26; // r13d
  int *v27; // rdi
  char *v28; // r14
  size_t v29; // rdx
  __int64 v30; // rdi
  char *v31; // r12
  int v33; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int8 *v34; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v35; // [rsp+30h] [rbp-80h]
  unsigned __int64 v36; // [rsp+38h] [rbp-78h]
  _DWORD v37[4]; // [rsp+40h] [rbp-70h] BYREF
  int *v38; // [rsp+50h] [rbp-60h] BYREF
  __int64 v39; // [rsp+58h] [rbp-58h]
  _BYTE v40[80]; // [rsp+60h] [rbp-50h] BYREF

  v6 = (const __m128i *)((char *)a5 + 8 * a6);
  v7 = (__int64 *)a5;
  v38 = (int *)v40;
  v39 = 0x100000000LL;
  v37[0] = (_DWORD)a2;
  v37[1] = a3;
  v37[2] = a4;
  if ( a5 != v6 )
  {
    do
    {
      v14 = *v7;
      v36 = 0;
      v34 = 0;
      v15 = *(unsigned __int8 **)(v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF));
      if ( sub_AC30F0((__int64)v15) )
        v35 = 0;
      else
        v35 = sub_BD3990(v15, (__int64)a2);
      v8 = *(_DWORD *)(v14 + 4) & 0x7FFFFFF;
      a2 = *(_BYTE **)(v14 + 32 * (1 - v8));
      v9 = (_QWORD *)*((_QWORD *)a2 + 3);
      if ( *((_DWORD *)a2 + 8) > 0x40u )
        v9 = (_QWORD *)*v9;
      v33 = (int)v9;
      v36 = *(_QWORD *)(v14 + 40) & 0xFFFFFFFFFFFFFFFBLL;
      v10 = sub_BD3990(*(unsigned __int8 **)(v14 + 32 * (2 - v8)), (__int64)a2);
      if ( *v10 == 60 )
        v34 = v10;
      v11 = (unsigned int)v39;
      a5 = (const __m128i *)&v33;
      v12 = v38;
      a6 = (unsigned int)v39 + 1LL;
      if ( a6 > HIDWORD(v39) )
      {
        if ( v38 > &v33 || &v33 >= &v38[8 * (unsigned int)v39] )
        {
          a2 = v40;
          sub_C8D5F0((__int64)&v38, v40, (unsigned int)v39 + 1LL, 0x20u, (__int64)&v33, a6);
          v12 = v38;
          v11 = (unsigned int)v39;
          a5 = (const __m128i *)&v33;
        }
        else
        {
          a2 = v40;
          v28 = (char *)((char *)&v33 - (char *)v38);
          sub_C8D5F0((__int64)&v38, v40, (unsigned int)v39 + 1LL, 0x20u, (__int64)&v33, a6);
          v12 = v38;
          v11 = (unsigned int)v39;
          a5 = (const __m128i *)&v28[(_QWORD)v38];
        }
      }
      ++v7;
      v13 = (__m128i *)&v12[8 * v11];
      *v13 = _mm_loadu_si128(a5);
      v13[1] = _mm_loadu_si128(a5 + 1);
      LODWORD(v39) = v39 + 1;
    }
    while ( v6 != (const __m128i *)v7 );
  }
  v16 = (const void **)v37;
  v17 = *(unsigned int *)(a1 + 248);
  v18 = *(unsigned int *)(a1 + 252);
  v19 = *(_QWORD *)(a1 + 240);
  v20 = v17 + 1;
  v21 = v17;
  if ( v17 + 1 > v18 )
  {
    v30 = a1 + 240;
    if ( v19 > (unsigned __int64)v37 || (unsigned __int64)v37 >= v19 + (v17 << 6) )
    {
      sub_3012B20(v30, v20, v19, v18, (__int64)a5, a6);
      v17 = *(unsigned int *)(a1 + 248);
      v19 = *(_QWORD *)(a1 + 240);
      v21 = *(_DWORD *)(a1 + 248);
    }
    else
    {
      v31 = (char *)v37 - v19;
      sub_3012B20(v30, v20, v19, v18, (__int64)a5, a6);
      v19 = *(_QWORD *)(a1 + 240);
      v17 = *(unsigned int *)(a1 + 248);
      v16 = (const void **)&v31[v19];
      v21 = *(_DWORD *)(a1 + 248);
    }
  }
  v22 = v19 + (v17 << 6);
  if ( v22 )
  {
    v23 = *((_DWORD *)v16 + 2);
    v24 = *v16;
    v25 = (void *)(v22 + 32);
    *(_QWORD *)(v22 + 16) = v22 + 32;
    *(_DWORD *)(v22 + 8) = v23;
    *(_QWORD *)v22 = v24;
    *(_QWORD *)(v22 + 24) = 0x100000000LL;
    v26 = *((_DWORD *)v16 + 6);
    if ( v26 && (const void **)(v22 + 16) != v16 + 2 )
    {
      v29 = 32;
      if ( v26 == 1
        || (sub_C8D5F0(v22 + 16, (const void *)(v22 + 32), v26, 0x20u, v22 + 16, v26),
            v25 = *(void **)(v22 + 16),
            (v29 = 32LL * *((unsigned int *)v16 + 6)) != 0) )
      {
        memcpy(v25, v16[2], v29);
      }
      *(_DWORD *)(v22 + 24) = v26;
    }
    v21 = *(_DWORD *)(a1 + 248);
  }
  v27 = v38;
  *(_DWORD *)(a1 + 248) = v21 + 1;
  if ( v27 != (int *)v40 )
    _libc_free((unsigned __int64)v27);
}
