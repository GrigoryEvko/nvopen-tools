// Function: sub_1178BF0
// Address: 0x1178bf0
//
_QWORD *__fastcall sub_1178BF0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  const char *v6; // rax
  __int64 v7; // rdx
  unsigned __int8 *v8; // rax
  __int64 v9; // r14
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  _QWORD *v13; // r12
  unsigned int **v15; // [rsp+0h] [rbp-80h]
  __int64 v16; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v17; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v18[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  v16 = a1[1];
  v15 = *(unsigned int ***)(*a1 + 32);
  v6 = sub_BD5D20(v16);
  v18[1] = v7;
  v19 = 261;
  v18[0] = v6;
  v8 = (unsigned __int8 *)sub_B36550(v15, a2, a3, a4, (__int64)v18, v16);
  v17 = v8;
  if ( *v8 > 0x1Cu )
    sub_B45260(v8, a1[1], 1);
  v9 = 0;
  v10 = (__int64 *)sub_B43CA0(a1[1]);
  v18[0] = *((_QWORD *)v17 + 1);
  v11 = sub_B6E160(v10, 0x192u, (__int64)v18, 1);
  v19 = 257;
  v12 = v11;
  if ( v11 )
    v9 = *(_QWORD *)(v11 + 24);
  v13 = sub_BD2CC0(88, 2u);
  if ( v13 )
  {
    sub_B44260((__int64)v13, **(_QWORD **)(v9 + 16), 56, 2u, 0, 0);
    v13[9] = 0;
    sub_B4A290((__int64)v13, v9, v12, (__int64 *)&v17, 1, (__int64)v18, 0, 0);
  }
  return v13;
}
