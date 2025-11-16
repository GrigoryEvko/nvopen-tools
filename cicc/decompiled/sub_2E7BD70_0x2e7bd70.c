// Function: sub_2E7BD70
// Address: 0x2e7bd70
//
unsigned __int64 __fastcall sub_2E7BD70(
        _QWORD *a1,
        unsigned __int16 a2,
        int a3,
        int a4,
        int a5,
        int a6,
        __int128 a7,
        __int64 a8,
        unsigned __int8 a9,
        int a10,
        int a11)
{
  int v13; // eax
  __m128i v14; // xmm0
  unsigned __int8 v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // r12
  __int64 v19; // rax
  int v20; // [rsp+0h] [rbp-60h]
  int v21; // [rsp+8h] [rbp-58h]
  int v22; // [rsp+20h] [rbp-40h]

  v13 = a8;
  v14 = _mm_loadu_si128((const __m128i *)&a7);
  v15 = a9;
  a1[26] += 88LL;
  v22 = v13;
  v16 = a1[16];
  v17 = (v16 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( a1[17] < v17 + 88 || !v16 )
  {
    v20 = a6;
    v21 = a5;
    v19 = sub_9D1E70((__int64)(a1 + 16), 88, 88, 4);
    a6 = v20;
    a5 = v21;
    v17 = v19;
    goto LABEL_4;
  }
  a1[16] = v17 + 88;
  if ( v17 )
LABEL_4:
    sub_2EAC440(v17, a2, a3, a4, a5, a6, v14.m128i_i32[0], v14.m128i_i32[2], v22, v15, a10, a11);
  return v17;
}
