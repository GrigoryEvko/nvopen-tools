// Function: sub_34081D0
// Address: 0x34081d0
//
unsigned __int8 **__fastcall sub_34081D0(
        unsigned __int8 **a1,
        _QWORD *a2,
        __int128 *a3,
        __int64 a4,
        unsigned int *a5,
        unsigned int *a6,
        __m128i a7)
{
  __int128 v10; // rax
  __int64 v11; // r9
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // rdx
  unsigned __int8 *v14; // r15
  unsigned __int8 *v15; // r14
  __int128 v16; // rax
  unsigned __int8 *v18; // rdx

  *(_QWORD *)&v10 = sub_3400D50((__int64)a2, 0, a4, 0, a7);
  v12 = sub_3406EB0(a2, 0x35u, a4, *a5, *((_QWORD *)a5 + 1), v11, *a3, v10);
  v14 = v13;
  v15 = v12;
  *(_QWORD *)&v16 = sub_3400D50((__int64)a2, 1, a4, 0, a7);
  a1[2] = sub_3406EB0(a2, 0x35u, a4, *a6, *((_QWORD *)a6 + 1), (__int64)a6, *a3, v16);
  *a1 = v15;
  a1[1] = v14;
  a1[3] = v18;
  return a1;
}
