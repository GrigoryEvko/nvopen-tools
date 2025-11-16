// Function: sub_343F2D0
// Address: 0x343f2d0
//
__int64 __fastcall sub_343F2D0(
        __int64 **a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        unsigned int a8)
{
  __int128 v13; // rax
  __int128 v14; // rax
  __int64 v15; // r9
  unsigned __int8 *v16; // rax
  int v17; // edx
  int v18; // edi
  unsigned __int8 *v19; // rdx
  __int64 *v20; // rax
  __m128i v21; // xmm0
  __int128 v23; // [rsp-20h] [rbp-80h]
  __int128 v24; // [rsp-20h] [rbp-80h]

  *(_QWORD *)&v13 = sub_3400E40(**a1, a8, *(_DWORD *)a1[1], a1[1][1], (__int64)a1[2], a7);
  *((_QWORD *)&v23 + 1) = a4;
  *(_QWORD *)&v23 = a3;
  *(_QWORD *)&v14 = sub_3406EB0(
                      (_QWORD *)**a1,
                      0xBEu,
                      (__int64)a1[2],
                      *(unsigned int *)a1[1],
                      a1[1][1],
                      *((__int64 *)&v13 + 1),
                      v23,
                      v13);
  *((_QWORD *)&v24 + 1) = a6;
  *(_QWORD *)&v24 = a5;
  v16 = sub_3406EB0((_QWORD *)**a1, a2, (__int64)a1[2], *(unsigned int *)a1[1], a1[1][1], v15, v24, v14);
  v18 = v17;
  v19 = v16;
  v20 = *a1;
  v21 = _mm_loadu_si128((const __m128i *)a1[3]);
  v20[2] = v21.m128i_i64[0];
  *((_DWORD *)v20 + 6) = v21.m128i_i32[2];
  v20[4] = (__int64)v19;
  *((_DWORD *)v20 + 10) = v18;
  return 1;
}
