// Function: sub_34092D0
// Address: 0x34092d0
//
unsigned __int8 *__fastcall sub_34092D0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        int a8)
{
  __int128 v9; // [rsp-20h] [rbp-30h]
  __int128 v10; // [rsp-10h] [rbp-20h]

  *((_QWORD *)&v10 + 1) = a5;
  *(_QWORD *)&v10 = a4;
  *((_QWORD *)&v9 + 1) = a3;
  *(_QWORD *)&v9 = a2;
  return sub_3405C90(
           a1,
           0x38u,
           a6,
           *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3),
           *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3 + 8),
           a8,
           a7,
           v9,
           v10);
}
