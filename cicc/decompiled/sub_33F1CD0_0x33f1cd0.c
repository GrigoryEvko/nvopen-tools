// Function: sub_33F1CD0
// Address: 0x33f1cd0
//
__m128i *__fastcall sub_33F1CD0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11,
        const __m128i *a12,
        char a13)
{
  unsigned __int16 *v15; // rax
  unsigned int v16; // ecx
  __int64 v17; // r8
  __int128 v18; // rax
  __int128 v20; // [rsp+0h] [rbp-70h]
  __int128 v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+30h] [rbp-40h] BYREF
  int v23; // [rsp+38h] [rbp-38h]

  *(_QWORD *)&v21 = a5;
  *((_QWORD *)&v21 + 1) = a6;
  v15 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  v22 = 0;
  v23 = 0;
  *(_QWORD *)&v18 = sub_33F17F0(a1, 51, (__int64)&v22, v16, v17);
  if ( v22 )
  {
    v20 = v18;
    sub_B91220((__int64)&v22, v22);
    v18 = v20;
  }
  return sub_33E8960(a1, 0, 0, a2, a3, a4, v21, a7, a8, v18, a9, a10, a11, a2, a3, a12, a13);
}
