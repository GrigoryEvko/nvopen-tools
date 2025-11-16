// Function: sub_1AB1D50
// Address: 0x1ab1d50
//
__int64 __fastcall sub_1AB1D50(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v15; // [rsp+20h] [rbp-70h] BYREF
  __int64 v16; // [rsp+28h] [rbp-68h]
  _QWORD v17[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v18[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 *v19[8]; // [rsp+50h] [rbp-40h] BYREF

  v6 = 0;
  v7 = *a4;
  v15 = a5;
  v16 = a6;
  if ( (*(_BYTE *)(v7 + 92) & 3) != 0 )
  {
    v10 = sub_157EB90(*(_QWORD *)(a3 + 8));
    v19[0] = (__int64 *)sub_16471D0(*(_QWORD **)(a3 + 24), 0);
    v19[1] = v19[0];
    v18[1] = 0x200000002LL;
    v11 = sub_1644EA0(v19[0], v19, 2, 0);
    v12 = sub_1632080(v10, v15, v16, v11, 0);
    sub_1AB1740(v10, v15, v16, a4);
    LOWORD(v19[0]) = 261;
    v18[0] = &v15;
    v17[0] = sub_1AB1800(a1, (__int64 *)a3);
    v17[1] = sub_1AB1800(a2, (__int64 *)a3);
    v6 = sub_1285290((__int64 *)a3, *(_QWORD *)(*(_QWORD *)v12 + 24LL), v12, (int)v17, 2, (__int64)v18, 0);
    v13 = sub_1649C60(v12);
    if ( !*(_BYTE *)(v13 + 16) )
      *(_WORD *)(v6 + 18) = *(_WORD *)(v6 + 18) & 0x8000 | *(_WORD *)(v6 + 18) & 3 | (*(_WORD *)(v13 + 18) >> 2) & 0xFFC;
  }
  return v6;
}
