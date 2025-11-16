// Function: sub_202E540
// Address: 0x202e540
//
__int64 *__fastcall sub_202E540(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  char *v9; // rdx
  __int64 *v10; // r13
  char v11; // al
  __int64 v12; // rdx
  unsigned int v13; // edx
  _BYTE v15[8]; // [rsp+0h] [rbp-20h] BYREF
  __int64 v16; // [rsp+8h] [rbp-18h]

  v9 = *(char **)(a2 + 40);
  v10 = *(__int64 **)(a1 + 8);
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v15[0] = v11;
  v16 = v12;
  if ( v11 )
    v13 = word_4305480[(unsigned __int8)(v11 - 14)];
  else
    v13 = sub_1F58D30((__int64)v15);
  return sub_1D40890(v10, a2, v13, a7, a8, a9, a3, a4, a5);
}
