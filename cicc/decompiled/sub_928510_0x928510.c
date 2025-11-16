// Function: sub_928510
// Address: 0x928510
//
__int64 __fastcall sub_928510(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // r12
  _BYTE *v12; // rsi
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-78h] BYREF
  __int64 v15; // [rsp+10h] [rbp-70h] BYREF
  _BYTE *v16; // [rsp+18h] [rbp-68h]
  _BYTE *v17; // [rsp+20h] [rbp-60h]
  _QWORD v18[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v19; // [rsp+50h] [rbp-30h]

  v14 = a2;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v5 = sub_B98A20(a2, a2, a3, a4);
  v6 = *(_QWORD *)(a1 + 40);
  v18[0] = v5;
  v7 = sub_B9C770(v6, v18, 1, 0, 1);
  v18[0] = sub_B9F6F0(*(_QWORD *)(a1 + 40), v7);
  sub_928380((__int64)&v15, 0, v18);
  v12 = v16;
  if ( v17 == v16 )
  {
    sub_9281F0((__int64)&v15, v16, &v14);
    v13 = v14;
  }
  else
  {
    v13 = v14;
    if ( v16 )
    {
      *(_QWORD *)v16 = v14;
      v12 = v16;
      v13 = v14;
    }
    v16 = v12 + 8;
  }
  v18[0] = *(_QWORD *)(v13 + 8);
  v8 = sub_B6E160(**(_QWORD **)(a1 + 32), 10578, v18, 1);
  v9 = 0;
  v19 = 257;
  if ( v8 )
    v9 = *(_QWORD *)(v8 + 24);
  v10 = sub_921880((unsigned int **)(a1 + 48), v9, v8, v15, (__int64)&v16[-v15] >> 3, (__int64)v18, 0);
  if ( v15 )
    j_j___libc_free_0(v15, &v17[-v15]);
  return v10;
}
