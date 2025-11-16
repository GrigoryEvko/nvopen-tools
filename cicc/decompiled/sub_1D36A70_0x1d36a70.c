// Function: sub_1D36A70
// Address: 0x1d36a70
//
void __fastcall sub_1D36A70(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        _QWORD *a6,
        __m128 a7,
        double a8,
        __m128i a9,
        _QWORD *a10)
{
  __int64 v10; // r12
  unsigned int v11; // r10d
  __int64 v14; // r14
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  const __m128i *v18; // r11
  __int64 v19; // rax
  const __m128i *v20; // r11
  __int64 v21; // rax
  _BYTE *v22; // rdx
  __int64 *v23; // rax
  unsigned int v24; // r10d
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 *v32; // rax
  __int128 v33; // [rsp-10h] [rbp-1A0h]
  __int64 v34; // [rsp+0h] [rbp-190h]
  __int64 v35; // [rsp+8h] [rbp-188h]
  const __m128i *v36; // [rsp+10h] [rbp-180h]
  const __m128i *v37; // [rsp+10h] [rbp-180h]
  __int64 v38; // [rsp+20h] [rbp-170h]
  unsigned int v40; // [rsp+30h] [rbp-160h]
  __int64 v41; // [rsp+30h] [rbp-160h]
  unsigned __int64 v43; // [rsp+48h] [rbp-148h]
  _BYTE *v44; // [rsp+50h] [rbp-140h] BYREF
  __int64 v45; // [rsp+58h] [rbp-138h]
  _BYTE v46[304]; // [rsp+60h] [rbp-130h] BYREF

  v11 = a4;
  v44 = v46;
  v45 = 0x1000000000LL;
  if ( a4 >= a5 )
  {
    v22 = v46;
    v21 = 0;
  }
  else
  {
    v14 = 16LL * a4;
    v15 = 16 * (a4 + (unsigned __int64)(a5 - 1 - a4) + 1);
    v16 = v15;
    do
    {
      v17 = *(unsigned int *)(a3 + 8);
      v18 = (const __m128i *)(v14 + *a6);
      if ( (unsigned int)v17 >= *(_DWORD *)(a3 + 12) )
      {
        v37 = (const __m128i *)(v14 + *a6);
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v15, (int)a6);
        v17 = *(unsigned int *)(a3 + 8);
        v18 = v37;
      }
      *(__m128i *)(*(_QWORD *)a3 + 16 * v17) = _mm_loadu_si128(v18);
      v19 = (unsigned int)v45;
      ++*(_DWORD *)(a3 + 8);
      v20 = (const __m128i *)(v14 + *a6);
      if ( (unsigned int)v19 >= HIDWORD(v45) )
      {
        v36 = (const __m128i *)(v14 + *a6);
        sub_16CD150((__int64)&v44, v46, 0, 16, v15, (int)a6);
        v19 = (unsigned int)v45;
        v20 = v36;
      }
      a7 = (__m128)_mm_loadu_si128(v20);
      v14 += 16;
      *(__m128 *)&v44[16 * v19] = a7;
      v21 = (unsigned int)(v45 + 1);
      LODWORD(v45) = v45 + 1;
    }
    while ( v16 != v14 );
    v11 = a4;
    v22 = v44;
  }
  *((_QWORD *)&v33 + 1) = v21;
  *(_QWORD *)&v33 = v22;
  v40 = v11;
  v23 = sub_1D359D0(a1, 2, a2, 1, 0, 0, *(double *)a7.m128_u64, a8, a9, v33);
  v24 = v40;
  v38 = v25;
  if ( v40 < a5 )
  {
    v41 = (__int64)v23;
    v26 = 16LL * v24;
    v43 = 16 * (v24 + (unsigned __int64)(a5 - 1 - v24) + 1);
    do
    {
      v27 = *(_QWORD *)(*a10 + v26);
      if ( *(_WORD *)(v27 + 24) != 186 )
        BUG();
      LOBYTE(v10) = *(_BYTE *)(v27 + 88);
      v29 = sub_1D2C2D0(
              a1,
              v41,
              v38,
              a2,
              *(_QWORD *)(*(_QWORD *)(v27 + 32) + 40LL),
              *(_QWORD *)(*(_QWORD *)(v27 + 32) + 48LL),
              *(_QWORD *)(*(_QWORD *)(v27 + 32) + 80LL),
              *(_QWORD *)(*(_QWORD *)(v27 + 32) + 88LL),
              v10,
              *(_QWORD *)(v27 + 96),
              *(_QWORD *)(v27 + 104));
      v30 = v28;
      v31 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v31 >= *(_DWORD *)(a3 + 12) )
      {
        v35 = v28;
        v34 = v29;
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v29, v28);
        v31 = *(unsigned int *)(a3 + 8);
        v29 = v34;
        v30 = v35;
      }
      v32 = (__int64 *)(*(_QWORD *)a3 + 16 * v31);
      v26 += 16;
      *v32 = v29;
      v32[1] = v30;
      ++*(_DWORD *)(a3 + 8);
    }
    while ( v43 != v26 );
  }
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
}
