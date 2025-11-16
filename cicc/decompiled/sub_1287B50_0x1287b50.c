// Function: sub_1287B50
// Address: 0x1287b50
//
__int64 __fastcall sub_1287B50(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  _BYTE *v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // [rsp+8h] [rbp-68h] BYREF
  __int64 v14; // [rsp+10h] [rbp-60h] BYREF
  _BYTE *v15; // [rsp+18h] [rbp-58h]
  _BYTE *v16; // [rsp+20h] [rbp-50h]
  _QWORD v17[2]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v18; // [rsp+40h] [rbp-30h]

  v13 = a2;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v5 = sub_1624210(a2, a2, a3, a4);
  v6 = *(_QWORD *)(a1 + 40);
  v17[0] = v5;
  v7 = sub_1627350(v6, v17, 1, 0, 1);
  v17[0] = sub_1628DA0(*(_QWORD *)(a1 + 40), v7);
  sub_12879C0((__int64)&v14, 0, v17);
  v11 = v15;
  if ( v16 == v15 )
  {
    sub_1287830((__int64)&v14, v15, &v13);
    v12 = v13;
  }
  else
  {
    v12 = v13;
    if ( v15 )
    {
      *(_QWORD *)v15 = v13;
      v11 = v15;
      v12 = v13;
    }
    v15 = v11 + 8;
  }
  v17[0] = *v12;
  v8 = sub_15E26F0(**(_QWORD **)(a1 + 32), 5232, v17, 1);
  v18 = 257;
  v9 = sub_1285290((__int64 *)(a1 + 48), *(_QWORD *)(v8 + 24), v8, v14, (__int64)&v15[-v14] >> 3, (__int64)v17, 0);
  if ( v14 )
    j_j___libc_free_0(v14, &v16[-v14]);
  return v9;
}
