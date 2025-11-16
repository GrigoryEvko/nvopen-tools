// Function: sub_1AB2190
// Address: 0x1ab2190
//
__int64 __fastcall sub_1AB2190(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // r15
  __int64 v12; // r15
  __int64 *v13; // rdi
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v19; // [rsp+0h] [rbp-A0h]
  _QWORD v21[4]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v22[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v23[10]; // [rsp+50h] [rbp-50h] BYREF

  v8 = 0;
  if ( (*(_BYTE *)(*a5 + 93) & 0xC0) != 0 )
  {
    v12 = sub_157EB90(*(_QWORD *)(a4 + 8));
    v13 = (__int64 *)sub_16471D0(*(_QWORD **)(a4 + 24), 0);
    v14 = a8;
    v23[2] = *a3;
    v19 = a7;
    v23[0] = v13;
    v23[1] = v13;
    v22[1] = 0x300000003LL;
    v15 = sub_1644EA0(v13, v23, 3, 0);
    v16 = sub_1632080(v12, v19, v14, v15, 0);
    sub_1AB1740(v12, a7, a8, a5);
    LOWORD(v23[0]) = 261;
    v22[0] = &a7;
    v21[0] = sub_1AB1800(a1, (__int64 *)a4);
    v21[2] = a3;
    v21[1] = sub_1AB1800(a2, (__int64 *)a4);
    v8 = sub_1285290((__int64 *)a4, *(_QWORD *)(*(_QWORD *)v16 + 24LL), v16, (int)v21, 3, (__int64)v22, 0);
    v17 = sub_1649C60(v16);
    if ( !*(_BYTE *)(v17 + 16) )
      *(_WORD *)(v8 + 18) = *(_WORD *)(v8 + 18) & 0x8000 | *(_WORD *)(v8 + 18) & 3 | (*(_WORD *)(v17 + 18) >> 2) & 0xFFC;
  }
  return v8;
}
