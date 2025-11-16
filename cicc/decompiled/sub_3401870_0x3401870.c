// Function: sub_3401870
// Address: 0x3401870
//
unsigned __int8 *__fastcall sub_3401870(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __int128 v12; // rax
  __int128 v14; // [rsp-50h] [rbp-A0h]
  __int128 v15; // [rsp-30h] [rbp-80h]

  *(_QWORD *)&v12 = sub_3401740((__int64)a1, 1, a2, (unsigned int)a8, *((__int64 *)&a8 + 1), a6, a8);
  *((_QWORD *)&v15 + 1) = a6;
  *(_QWORD *)&v15 = a5;
  *((_QWORD *)&v14 + 1) = a4;
  *(_QWORD *)&v14 = a3;
  return sub_33FC130(a1, 407, a2, (unsigned int)a8, *((__int64 *)&a8 + 1), a8, v14, v12, v15, a7);
}
