// Function: sub_3402E70
// Address: 0x3402e70
//
unsigned __int8 *__fastcall sub_3402E70(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int128 v7; // [rsp-10h] [rbp-10h]

  *((_QWORD *)&v7 + 1) = *(unsigned int *)(a3 + 8);
  *(_QWORD *)&v7 = *(_QWORD *)a3;
  return sub_33FC220(a1, 2, a2, 1, 0, a6, v7);
}
