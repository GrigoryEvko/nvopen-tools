// Function: sub_20A0A00
// Address: 0x20a0a00
//
__int64 *__fastcall sub_20A0A00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        __m128i a8)
{
  __int64 v11; // r14
  unsigned int v12; // edx
  unsigned __int64 v13; // r15
  __int64 v14; // r12
  unsigned int v15; // edx
  unsigned __int64 v16; // r13
  __int64 *v17; // rax
  unsigned int v18; // edx
  __int128 v20; // [rsp-20h] [rbp-90h]
  __int128 v21; // [rsp-10h] [rbp-80h]
  __int128 v22; // [rsp-10h] [rbp-80h]

  *((_QWORD *)&v21 + 1) = a3;
  *(_QWORD *)&v21 = a2;
  v11 = sub_1D309E0(
          *(__int64 **)a1,
          143,
          *(_QWORD *)(a1 + 8),
          **(unsigned int **)(a1 + 16),
          *(const void ***)(*(_QWORD *)(a1 + 16) + 8LL),
          0,
          a6,
          a7,
          *(double *)a8.m128i_i64,
          v21);
  v13 = v12 | a3 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v20 + 1) = a5;
  *(_QWORD *)&v20 = a4;
  v14 = sub_1D309E0(
          *(__int64 **)a1,
          143,
          *(_QWORD *)(a1 + 8),
          **(unsigned int **)(a1 + 16),
          *(const void ***)(*(_QWORD *)(a1 + 16) + 8LL),
          0,
          a6,
          a7,
          *(double *)a8.m128i_i64,
          v20);
  v16 = v15 | a5 & 0xFFFFFFFF00000000LL;
  v17 = sub_1D332F0(
          *(__int64 **)a1,
          122,
          *(_QWORD *)(a1 + 8),
          **(unsigned int **)(a1 + 16),
          *(const void ***)(*(_QWORD *)(a1 + 16) + 8LL),
          0,
          a6,
          a7,
          a8,
          v14,
          v16,
          *(_OWORD *)*(_QWORD *)(a1 + 24));
  *((_QWORD *)&v22 + 1) = v18 | v16 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v22 = v17;
  return sub_1D332F0(
           *(__int64 **)a1,
           119,
           *(_QWORD *)(a1 + 8),
           **(unsigned int **)(a1 + 16),
           *(const void ***)(*(_QWORD *)(a1 + 16) + 8LL),
           0,
           a6,
           a7,
           a8,
           v11,
           v13,
           v22);
}
