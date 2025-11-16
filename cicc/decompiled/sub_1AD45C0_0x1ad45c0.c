// Function: sub_1AD45C0
// Address: 0x1ad45c0
//
_QWORD *__fastcall sub_1AD45C0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  int v11; // r8d
  __int64 *v12; // rcx
  __int64 *v13; // rdx
  int v14; // esi
  __int64 v15; // rax
  _QWORD *v16; // r13
  __int64 *v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax
  int v20; // r8d
  __int64 v21; // r9
  _QWORD *v23; // rax
  __int64 *v24; // [rsp+0h] [rbp-50h]
  int v25; // [rsp+Ch] [rbp-44h]

  v11 = a4;
  v12 = &a5[7 * a6];
  if ( v12 == a5 )
  {
    v23 = sub_1648AB0(72, (int)a4 + 1, 16 * (int)a6);
    v20 = a4 + 1;
    v21 = a4;
    v16 = v23;
    if ( !v23 )
      return v16;
    goto LABEL_8;
  }
  v13 = a5;
  v14 = 0;
  do
  {
    v15 = v13[5] - v13[4];
    v13 += 7;
    v14 += v15 >> 3;
  }
  while ( v12 != v13 );
  v24 = &a5[7 * a6];
  v25 = v11 + 1;
  v16 = sub_1648AB0(72, v11 + 1 + v14, 16 * (int)a6);
  if ( v16 )
  {
    v17 = a5;
    LODWORD(v18) = 0;
    do
    {
      v19 = v17[5] - v17[4];
      v17 += 7;
      v18 = (unsigned int)(v19 >> 3) + (unsigned int)v18;
    }
    while ( v24 != v17 );
    v20 = v18 + v25;
    v21 = a4 + v18;
LABEL_8:
    sub_15F1EA0((__int64)v16, **(_QWORD **)(a1 + 16), 54, (__int64)&v16[-3 * v21 - 3], v20, a8);
    v16[7] = 0;
    sub_15F5B40((__int64)v16, a1, a2, a3, a4, a7, a5, a6);
  }
  return v16;
}
