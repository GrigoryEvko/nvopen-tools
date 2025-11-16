// Function: sub_1D24450
// Address: 0x1d24450
//
__int64 *__fastcall sub_1D24450(
        _QWORD *a1,
        unsigned __int16 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 *v11; // r15
  __int64 v12; // rsi
  __int64 *v13; // rbx
  __int64 v14; // rsi
  int v15; // eax
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 *v21; // rbx
  int v23; // r12d
  __int128 v24; // [rsp-20h] [rbp-130h]
  __int64 *v29; // [rsp+48h] [rbp-C8h] BYREF
  unsigned __int64 v30[2]; // [rsp+50h] [rbp-C0h] BYREF
  _BYTE v31[176]; // [rsp+60h] [rbp-B0h] BYREF

  v11 = a9;
  v12 = a4;
  if ( !a4 )
    v12 = a5;
  v30[0] = (unsigned __int64)v31;
  v30[1] = 0x2000000000LL;
  v13 = &a9[2 * a10];
  sub_16BD4D0((__int64)v30, v12);
  sub_16BD430((__int64)v30, a2);
  sub_16BD4C0((__int64)v30, a7);
  if ( v13 != a9 )
  {
    do
    {
      v14 = *v11;
      v11 += 2;
      sub_16BD4C0((__int64)v30, v14);
      sub_16BD430((__int64)v30, *((_DWORD *)v11 - 2));
    }
    while ( v13 != v11 );
  }
  v15 = sub_1E340A0(a6);
  sub_16BD430((__int64)v30, v15);
  v29 = 0;
  v16 = sub_1D17920((__int64)a1, (__int64)v30, a3, (__int64 *)&v29);
  v21 = v16;
  if ( v16 )
  {
    sub_1E34340(v16[13], a6, v17, v18, v19, v20, a10, a9, a8);
  }
  else
  {
    v21 = (__int64 *)a1[26];
    v23 = *(_DWORD *)(a3 + 8);
    if ( v21 )
      a1[26] = *v21;
    else
      v21 = (__int64 *)sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v24 + 1) = a5;
    *(_QWORD *)&v24 = a4;
    sub_1D189E0((__int64)v21, a2, v23, (unsigned __int8 **)a3, a7, a8, v24, a6);
    sub_1D23B60((__int64)a1, (__int64)v21, (__int64)a9, a10);
    sub_16BDA20(a1 + 40, v21, v29);
    sub_1D172A0((__int64)a1, (__int64)v21);
  }
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
  return v21;
}
