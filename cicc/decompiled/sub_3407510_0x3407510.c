// Function: sub_3407510
// Address: 0x3407510
//
unsigned __int8 *__fastcall sub_3407510(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int128 v9; // rax
  __int128 v11; // [rsp-30h] [rbp-70h]
  __int128 v12; // [rsp-10h] [rbp-50h]

  *((_QWORD *)&v12 + 1) = a6;
  *(_QWORD *)&v12 = a5;
  *(_QWORD *)&v9 = sub_3401740((__int64)a1, 1, a2, (unsigned int)a5, a6, a6, v12);
  *((_QWORD *)&v11 + 1) = a4;
  *(_QWORD *)&v11 = a3;
  return sub_3406EB0(a1, 0xBCu, a2, (unsigned int)a5, a6, a5, v11, v9);
}
