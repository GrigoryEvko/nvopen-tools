// Function: sub_289AF80
// Address: 0x289af80
//
void __fastcall sub_289AF80(
        __int64 a1,
        const __m128i *a2,
        __int64 a3,
        __int16 a4,
        char a5,
        unsigned int a6,
        unsigned int a7,
        char a8,
        _BYTE *a9,
        _BYTE *a10,
        __int64 *a11,
        __int64 a12)
{
  _QWORD *v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  __int64 v19; // r8
  __int32 v20; // esi
  __int32 v21; // eax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // r8
  __m128i v28; // xmm0
  __int64 v29; // [rsp+8h] [rbp-1C8h]
  __int64 v31; // [rsp+10h] [rbp-1C0h]
  _BYTE *v35; // [rsp+38h] [rbp-198h] BYREF
  __m128i v36; // [rsp+40h] [rbp-190h] BYREF
  char v37; // [rsp+50h] [rbp-180h] BYREF
  __int16 v38; // [rsp+60h] [rbp-170h]
  unsigned __int64 v39[2]; // [rsp+F0h] [rbp-E0h] BYREF
  _BYTE v40[16]; // [rsp+100h] [rbp-D0h] BYREF
  __int16 v41; // [rsp+110h] [rbp-C0h]
  __m128i v42; // [rsp+180h] [rbp-50h]
  __int8 v43; // [rsp+190h] [rbp-40h]

  if ( !a8 )
    a6 = a7;
  v13 = *(_QWORD **)(a12 + 72);
  v14 = a6;
  v38 = 257;
  v41 = 257;
  v15 = sub_BCB2E0(v13);
  v16 = (_BYTE *)sub_ACD640(v15, v14, 0);
  v17 = (_BYTE *)sub_A81850((unsigned int **)a12, a10, v16, (__int64)v39, 0, 0);
  v18 = (_BYTE *)sub_929C50((unsigned int **)a12, v17, a9, (__int64)&v36, 0, 0);
  v41 = 257;
  v35 = v18;
  v19 = sub_921130((unsigned int **)a12, (__int64)a11, a3, &v35, 1, (__int64)v39, 0);
  if ( a2[10].m128i_i8[0] )
  {
    v20 = a2->m128i_i32[2];
    v21 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a2->m128i_i64[0] + 8LL) + 32LL);
  }
  else
  {
    v21 = a2->m128i_i32[2];
    v20 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a2->m128i_i64[0] + 8LL) + 32LL);
  }
  v31 = v19;
  v22 = sub_BCDA70(a11, v21 * v20);
  v23 = sub_BCB2E0(*(_QWORD **)(a12 + 72));
  v24 = sub_ACD640(v23, v14, 0);
  v26 = a2->m128i_u32[2];
  v39[0] = (unsigned __int64)v40;
  v39[1] = 0x1000000000LL;
  v27 = v31;
  if ( (_DWORD)v26 )
  {
    v29 = v24;
    sub_2894AD0((__int64)v39, (__int64)a2, v26, 0x1000000000LL, v31, v25);
    v24 = v29;
    v27 = v31;
  }
  v28 = _mm_loadu_si128(a2 + 9);
  v43 = a2[10].m128i_i8[0];
  v42 = v28;
  sub_289A510(&v36, a1, v22, (__int64)v39, v27, a4, v24, a5, a12);
  if ( (char *)v36.m128i_i64[0] != &v37 )
    _libc_free(v36.m128i_u64[0]);
  if ( (_BYTE *)v39[0] != v40 )
    _libc_free(v39[0]);
}
