// Function: sub_21453A0
// Address: 0x21453a0
//
void __fastcall sub_21453A0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned int a4,
        __int64 a5,
        double a6,
        double a7,
        __m128i a8,
        __int64 a9,
        __int64 a10,
        const void **a11)
{
  __int64 v15; // rcx
  __int64 v16; // rsi
  int v17; // eax
  unsigned int v18; // ebx
  int v19; // r9d
  int v20; // r9d
  __m128i v21; // xmm0
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // r14
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // rdx
  __int64 *v28; // rdx
  __int128 v29; // [rsp-10h] [rbp-90h]
  __int64 v30; // [rsp+20h] [rbp-60h] BYREF
  int v31; // [rsp+28h] [rbp-58h]
  __m128i v32; // [rsp+30h] [rbp-50h] BYREF
  __int128 v33; // [rsp+40h] [rbp-40h] BYREF

  v15 = a2;
  v16 = *(_QWORD *)(a2 + 72);
  v30 = v16;
  if ( v16 )
  {
    sub_1623A60((__int64)&v30, v16, 2);
    v15 = a2;
  }
  v17 = *(_DWORD *)(v15 + 64);
  v32 = 0;
  v31 = v17;
  v33 = 0;
  if ( a4 <= 1 )
  {
    *((_QWORD *)&v29 + 1) = a3;
    *(_QWORD *)&v29 = a2;
    v24 = sub_1D309E0(*(__int64 **)(a1 + 8), 158, (__int64)&v30, a10, a11, 0, 0.0, a7, *(double *)a8.m128i_i64, v29);
    v26 = v25;
    v27 = *(unsigned int *)(a5 + 8);
    if ( (unsigned int)v27 >= *(_DWORD *)(a5 + 12) )
    {
      sub_16CD150(a5, (const void *)(a5 + 16), 0, 16, v22, v23);
      v27 = *(unsigned int *)(a5 + 8);
    }
    v28 = (__int64 *)(*(_QWORD *)a5 + 16 * v27);
    *v28 = v24;
    v28[1] = v26;
    ++*(_DWORD *)(a5 + 8);
  }
  else
  {
    v18 = a4 >> 1;
    sub_200E870(a1, a2, a3, (__int64)&v32, &v33, (__m128i)0LL, a7, a8);
    if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) )
    {
      v21 = _mm_load_si128(&v32);
      v32.m128i_i64[0] = v33;
      v32.m128i_i32[2] = DWORD2(v33);
      *(_QWORD *)&v33 = v21.m128i_i64[0];
      DWORD2(v33) = v21.m128i_i32[2];
    }
    sub_21453A0(a1, v32.m128i_i32[0], v32.m128i_i32[2], v18, a5, v19, a10, (__int64)a11);
    sub_21453A0(a1, v33, DWORD2(v33), v18, a5, v20, a10, (__int64)a11);
  }
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
}
