// Function: sub_3466750
// Address: 0x3466750
//
unsigned __int8 *__fastcall sub_3466750(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int128 a8)
{
  unsigned __int16 v10; // ax
  unsigned __int16 v11; // ax
  int v12; // ecx
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 *v17; // [rsp+10h] [rbp-50h]
  unsigned int v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  __int64 v20; // [rsp+28h] [rbp-38h]

  v19 = a5;
  v20 = a6;
  if ( (_WORD)a5 )
  {
    v16 = 0;
    v10 = word_4456580[(unsigned __int16)a5 - 1];
  }
  else
  {
    v10 = sub_3009970((__int64)&v19, (__int64)a2, a3, a4, a5);
    v16 = v14;
  }
  v17 = (__int64 *)a2[8];
  v18 = v10;
  v11 = sub_2D43050(v10, 1);
  v12 = 0;
  if ( !v11 )
  {
    v11 = sub_3009400(v17, v18, v16, 1, 0);
    v12 = v15;
  }
  return sub_3465D80(a7, a1, a2, a3, a4, (unsigned int)v19, v20, v11, v12, a8);
}
