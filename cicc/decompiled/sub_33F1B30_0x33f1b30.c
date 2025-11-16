// Function: sub_33F1B30
// Address: 0x33f1b30
//
__m128i *__fastcall sub_33F1B30(
        __int64 *a1,
        char a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        const __m128i *a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  unsigned __int16 *v14; // rax
  unsigned int v15; // ecx
  __int64 v16; // r8
  __int128 v17; // rax
  __int128 v19; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  int v22; // [rsp+28h] [rbp-38h]

  v14 = (unsigned __int16 *)(*(_QWORD *)(a8 + 48) + 16LL * (unsigned int)a9);
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v21 = 0;
  v22 = 0;
  *(_QWORD *)&v17 = sub_33F17F0(a1, 51, (__int64)&v21, v15, v16);
  if ( v21 )
  {
    v19 = v17;
    sub_B91220((__int64)&v21, v21);
    v17 = v19;
  }
  return sub_33E9C90(a1, 0, a2, a4, a5, a3, a7, a8, a9, v17, a10, a11, a6);
}
