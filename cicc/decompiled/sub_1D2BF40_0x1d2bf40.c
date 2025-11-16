// Function: sub_1D2BF40
// Address: 0x1d2bf40
//
__int64 __fastcall sub_1D2BF40(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int64 a10,
        unsigned int a11,
        unsigned int a12,
        __int64 a13)
{
  __int64 v13; // rax
  __int64 v15; // r14
  __int64 v17; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r10
  __int64 v23; // rax
  char v24; // di
  __int64 v25; // rax
  int v26; // edx
  int v27; // r10d
  __int64 v28; // rax
  int v30; // eax
  unsigned __int8 *v31; // rax
  unsigned int v32; // eax
  __m128i v33; // xmm0
  int v34; // [rsp+8h] [rbp-78h]
  unsigned int v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  unsigned __int16 v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+18h] [rbp-68h]
  __m128i v40; // [rsp+20h] [rbp-60h] BYREF
  __int64 v41; // [rsp+30h] [rbp-50h]
  char v42[8]; // [rsp+40h] [rbp-40h] BYREF
  __int64 v43; // [rsp+48h] [rbp-38h]

  v13 = 16LL * (unsigned int)a6;
  v15 = a2;
  v17 = v13;
  v20 = a11;
  v21 = a12;
  if ( !a11 )
  {
    v39 = v13;
    v31 = (unsigned __int8 *)(*(_QWORD *)(a5 + 40) + v13);
    a2 = *v31;
    v35 = a12;
    v32 = sub_1D172F0((__int64)a1, a2, *((_QWORD *)v31 + 1));
    v21 = v35;
    v17 = v39;
    a11 = v32;
  }
  v38 = v21 | 2;
  if ( (a9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    a2 = (__int64)&a9;
    v36 = v17;
    sub_1D13370(&v40, (const __m128i *)&a9, (__int64)a1, a7, 0);
    v33 = _mm_loadu_si128(&v40);
    v17 = v36;
    a10 = v41;
    a9 = (__int128)v33;
  }
  v22 = a1[4];
  v23 = v17 + *(_QWORD *)(a5 + 40);
  v24 = *(_BYTE *)v23;
  v25 = *(_QWORD *)(v23 + 8);
  v42[0] = v24;
  v43 = v25;
  if ( v24 )
  {
    v26 = sub_1D13440(v24);
  }
  else
  {
    v34 = v22;
    v30 = sub_1F58D40(v42, a2, v20, v21, a5, v17);
    v27 = v34;
    v26 = v30;
  }
  v28 = sub_1E0B8E0(v27, v38, (unsigned int)(v26 + 7) >> 3, a11, a13, 0, a9, a10, 1, 0, 0);
  return sub_1D2BB40(a1, v15, a3, a4, a5, a6, a7, a8, v28);
}
