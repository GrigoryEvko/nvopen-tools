// Function: sub_3402B60
// Address: 0x3402b60
//
void __fastcall sub_3402B60(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        _QWORD *a6,
        _QWORD *a7)
{
  unsigned __int64 v7; // r14
  __int64 v8; // r15
  __int64 v10; // r8
  __int64 v11; // r15
  __int64 v12; // rbx
  __m128i v14; // xmm0
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __m128i v17; // xmm0
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // r13
  __int64 v22; // r14
  unsigned __int8 *v23; // rax
  const void *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __m128i *v27; // r8
  __int64 v28; // rax
  __int64 v29; // r9
  __m128i **v30; // rax
  __int128 v31; // [rsp-10h] [rbp-1A0h]
  __m128i v32; // [rsp+0h] [rbp-190h] BYREF
  _BYTE *v33; // [rsp+18h] [rbp-178h]
  unsigned __int64 v34; // [rsp+20h] [rbp-170h]
  __int64 v35; // [rsp+28h] [rbp-168h]
  const void *v36; // [rsp+30h] [rbp-160h]
  __int64 v37; // [rsp+38h] [rbp-158h]
  _QWORD *v38; // [rsp+40h] [rbp-150h]
  __int64 v39; // [rsp+48h] [rbp-148h]
  _BYTE *v40; // [rsp+50h] [rbp-140h] BYREF
  __int64 v41; // [rsp+58h] [rbp-138h]
  _BYTE v42[304]; // [rsp+60h] [rbp-130h] BYREF

  v33 = v42;
  v40 = v42;
  v38 = a1;
  v37 = a2;
  v41 = 0x1000000000LL;
  if ( a4 >= a5 )
  {
    sub_33FC220(v38, 2, v37, 1, 0, (__int64)a6, (unsigned __int64)v33);
  }
  else
  {
    v34 = v7;
    v10 = a5 - 1 - a4;
    v35 = v8;
    v36 = (const void *)(a3 + 16);
    v39 = 16LL * a4;
    v11 = 16 * (a4 + v10 + 1);
    v12 = v39;
    do
    {
      v14 = _mm_loadu_si128((const __m128i *)(*a6 + v12));
      v15 = *(unsigned int *)(a3 + 8);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v32 = v14;
        sub_C8D5F0(a3, v36, v15 + 1, 0x10u, v10, (__int64)a6);
        v15 = *(unsigned int *)(a3 + 8);
        v14 = _mm_load_si128(&v32);
      }
      *(__m128i *)(*(_QWORD *)a3 + 16 * v15) = v14;
      v16 = HIDWORD(v41);
      ++*(_DWORD *)(a3 + 8);
      v17 = _mm_loadu_si128((const __m128i *)(*a6 + v12));
      v18 = (unsigned int)v41;
      v19 = (unsigned int)v41 + 1LL;
      if ( v19 > v16 )
      {
        v32 = v17;
        sub_C8D5F0((__int64)&v40, v33, v19, 0x10u, v10, (__int64)a6);
        v18 = (unsigned int)v41;
        v17 = _mm_load_si128(&v32);
      }
      v12 += 16;
      *(__m128i *)&v40[16 * v18] = v17;
      v20 = v41 + 1;
      LODWORD(v41) = v41 + 1;
    }
    while ( v11 != v12 );
    v21 = v39;
    *((_QWORD *)&v31 + 1) = v20;
    v22 = v34;
    *(_QWORD *)&v31 = v40;
    v23 = sub_33FC220(v38, 2, v37, 1, 0, (__int64)a6, v31);
    v39 = v11;
    v36 = v24;
    v34 = (unsigned __int64)v23;
    do
    {
      v25 = *(_QWORD *)(*a7 + v21);
      if ( *(_DWORD *)(v25 + 24) != 299 )
        BUG();
      LOWORD(v22) = *(_WORD *)(v25 + 96);
      v27 = sub_33F49B0(
              v38,
              v34,
              (unsigned __int64)v36,
              v37,
              *(_QWORD *)(*(_QWORD *)(v25 + 40) + 40LL),
              *(_QWORD *)(*(_QWORD *)(v25 + 40) + 48LL),
              *(_QWORD *)(*(_QWORD *)(v25 + 40) + 80LL),
              *(_QWORD *)(*(_QWORD *)(v25 + 40) + 88LL),
              v22,
              *(_QWORD *)(v25 + 104),
              *(const __m128i **)(v25 + 112));
      v28 = *(unsigned int *)(a3 + 8);
      v29 = v26;
      if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v32.m128i_i64[0] = (__int64)v27;
        v32.m128i_i64[1] = v26;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v28 + 1, 0x10u, (__int64)v27, v26);
        v28 = *(unsigned int *)(a3 + 8);
        v29 = v32.m128i_i64[1];
        v27 = (__m128i *)v32.m128i_i64[0];
      }
      v30 = (__m128i **)(*(_QWORD *)a3 + 16 * v28);
      v21 += 16;
      *v30 = v27;
      v30[1] = (__m128i *)v29;
      ++*(_DWORD *)(a3 + 8);
    }
    while ( v39 != v21 );
  }
  if ( v40 != v33 )
    _libc_free((unsigned __int64)v40);
}
