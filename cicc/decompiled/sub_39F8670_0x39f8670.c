// Function: sub_39F8670
// Address: 0x39f8670
//
void __fastcall __spoils<rdx,rcx,r8,r9,r10,r11,xmm0,xmm4,xmm5> sub_39F8670(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  __m128i v9; // xmm3
  __m128i v10; // xmm4
  __m128i v11; // xmm5
  __m128i v12; // xmm6
  __m128i v13; // xmm7
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  bool v19; // zf
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // [rsp+0h] [rbp-228h] BYREF
  __m128i v26; // [rsp+8h] [rbp-220h] BYREF
  __m128i v27; // [rsp+18h] [rbp-210h] BYREF
  __m128i v28; // [rsp+28h] [rbp-200h] BYREF
  __m128i v29; // [rsp+38h] [rbp-1F0h] BYREF
  __m128i v30; // [rsp+48h] [rbp-1E0h] BYREF
  __m128i v31; // [rsp+58h] [rbp-1D0h] BYREF
  __m128i v32; // [rsp+68h] [rbp-1C0h] BYREF
  __m128i v33; // [rsp+78h] [rbp-1B0h] BYREF
  __m128i v34; // [rsp+88h] [rbp-1A0h] BYREF
  __m128i v35; // [rsp+98h] [rbp-190h] BYREF
  __m128i v36; // [rsp+A8h] [rbp-180h] BYREF
  __m128i v37; // [rsp+B8h] [rbp-170h] BYREF
  __m128i v38; // [rsp+C8h] [rbp-160h] BYREF
  __m128i v39; // [rsp+D8h] [rbp-150h] BYREF
  __m128i v40; // [rsp+E8h] [rbp-140h] BYREF
  __m128i v41[15]; // [rsp+F8h] [rbp-130h] BYREF
  __int64 v42; // [rsp+1F8h] [rbp-30h]
  __int64 retaddr; // [rsp+230h] [rbp+8h]

  v42 = a3;
  sub_39F7A80(&v26, (__int64)&a7, retaddr);
  v7 = _mm_loadu_si128(&v27);
  v8 = _mm_loadu_si128(&v28);
  v9 = _mm_loadu_si128(&v29);
  v10 = _mm_loadu_si128(&v30);
  v11 = _mm_loadu_si128(&v31);
  v41[0] = _mm_loadu_si128(&v26);
  v12 = _mm_loadu_si128(&v32);
  v13 = _mm_loadu_si128(&v33);
  v41[1] = v7;
  v14 = _mm_loadu_si128(&v34);
  v15 = _mm_loadu_si128(&v35);
  v41[2] = v8;
  v41[3] = v9;
  v16 = _mm_loadu_si128(&v36);
  v17 = _mm_loadu_si128(&v37);
  v41[4] = v10;
  v18 = _mm_loadu_si128(&v38);
  v19 = a1[2] == 0;
  v41[5] = v11;
  v20 = _mm_loadu_si128(&v39);
  v41[6] = v12;
  v21 = _mm_loadu_si128(&v40);
  v41[7] = v13;
  v41[8] = v14;
  v41[9] = v15;
  v41[10] = v16;
  v41[11] = v17;
  v41[12] = v18;
  v41[13] = v20;
  v41[14] = v21;
  if ( v19 )
    v22 = sub_39F7C00(a1, v41, &v25);
  else
    v22 = sub_39F7D50(a1, v41, &v25);
  if ( v22 != 7 )
    abort();
  sub_39F5CF0((__int64)&v26, (__int64)v41);
  nullsub_2004();
  *(__int64 *)((char *)&retaddr + v23) = v24;
}
