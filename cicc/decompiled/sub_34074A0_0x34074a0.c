// Function: sub_34074A0
// Address: 0x34074a0
//
unsigned __int8 *__fastcall sub_34074A0(
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
  __int128 v13; // [rsp-20h] [rbp-60h]

  *(_QWORD *)&v10 = sub_34015B0((__int64)a1, a2, a5, a6, 0, 0, a7);
  *((_QWORD *)&v13 + 1) = a4;
  *(_QWORD *)&v13 = a3;
  return sub_3406EB0(a1, 0xBCu, a2, a5, a6, v11, v13, v10);
}
