// Function: sub_1101320
// Address: 0x1101320
//
unsigned __int8 *__fastcall sub_1101320(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  char *v6; // rbx
  __int64 v7; // r14
  _BYTE v9[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v10; // [rsp+20h] [rbp-30h]

  v6 = *(char **)(a2 - 32);
  if ( (unsigned __int8)(*v6 - 72) > 1u )
    return sub_11005E0(a1, (unsigned __int8 *)a2, a3, a4, a5, a6);
  v7 = *(_QWORD *)(a2 + 8);
  if ( !sub_10FD370(v6, a1) )
    return sub_11005E0(a1, (unsigned __int8 *)a2, a3, a4, a5, a6);
  v10 = 257;
  return (unsigned __int8 *)sub_B51D30(
                              (unsigned int)(unsigned __int8)*v6 - 29,
                              *((_QWORD *)v6 - 4),
                              v7,
                              (__int64)v9,
                              0,
                              0);
}
