// Function: sub_3407430
// Address: 0x3407430
//
unsigned __int8 *__fastcall sub_3407430(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __m128i a7)
{
  __int128 v10; // rax
  __int64 v11; // r9
  __int128 v13; // [rsp-18h] [rbp-60h]

  *(_QWORD *)&v10 = sub_3400BD0((__int64)a1, 0, a4, a5, a6, 0, a7, 0);
  *((_QWORD *)&v13 + 1) = a3;
  *(_QWORD *)&v13 = a2;
  return sub_3406EB0(a1, 0x39u, a4, a5, a6, v11, v10, v13);
}
