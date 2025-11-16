// Function: sub_24F3110
// Address: 0x24f3110
//
_QWORD *__fastcall sub_24F3110(__int64 **a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r13
  _QWORD *v11; // r12
  __int64 v13[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v14[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  v5 = a4 + 24;
  v6 = sub_BCB2B0(a1[1]);
  v7 = sub_ACD640(v6, a3, 0);
  v8 = sub_B6E160(*a1, 0x3Bu, 0, 0);
  v13[0] = a2;
  v9 = 0;
  v15 = 257;
  v10 = v8;
  v13[1] = v7;
  if ( v8 )
    v9 = *(_QWORD *)(v8 + 24);
  v11 = sub_BD2C40(88, 3u);
  if ( v11 )
  {
    sub_B44260((__int64)v11, **(_QWORD **)(v9 + 16), 56, 3u, v5, 0);
    v11[9] = 0;
    sub_B4A290((__int64)v11, v9, v10, v13, 2, (__int64)v14, 0, 0);
  }
  return v11;
}
