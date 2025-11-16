// Function: sub_343F220
// Address: 0x343f220
//
unsigned __int8 *__fastcall sub_343F220(
        __m128i a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        _QWORD *a9)
{
  __int64 v9; // r10
  __int64 v10; // r11
  unsigned __int64 v11; // r13
  __int64 v12; // r12
  _QWORD *v13; // rdi
  unsigned __int8 *v15; // rax
  unsigned int v16; // edx
  __int128 v17; // [rsp-20h] [rbp-60h]
  __int128 v18; // [rsp-10h] [rbp-50h]

  v9 = a6;
  v10 = a7;
  v11 = a5;
  v12 = a4;
  v13 = a9;
  if ( *(_DWORD *)(*a9 + 564LL) == 1 )
  {
    v15 = sub_3407580((__int64)a9, a8, a4, a5, a3, a1);
    v9 = a6;
    v10 = a7;
    v12 = (__int64)v15;
    v13 = a9;
    v11 = v16 | v11 & 0xFFFFFFFF00000000LL;
  }
  *((_QWORD *)&v18 + 1) = v10;
  *(_QWORD *)&v18 = v9;
  *((_QWORD *)&v17 + 1) = v11;
  *(_QWORD *)&v17 = v12;
  return sub_3406EB0(v13, 0x12Eu, a3, 1, 0, a7, v17, v18);
}
