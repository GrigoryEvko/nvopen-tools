// Function: sub_2464F50
// Address: 0x2464f50
//
__int64 __fastcall sub_2464F50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  unsigned __int64 v14; // r14
  _QWORD *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // [rsp+0h] [rbp-80h] BYREF
  __int64 v20; // [rsp+8h] [rbp-78h]
  _QWORD v21[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v22[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v23; // [rsp+40h] [rbp-40h]

  v22[0] = sub_9208B0(a5, a3);
  v22[1] = v8;
  v19 = (unsigned __int64)(v22[0] + 7LL) >> 3;
  LOBYTE(v20) = v8;
  v9 = sub_B33F60(a4, *(_QWORD *)(a1[1] + 80), v19, v8);
  v10 = a1[1];
  if ( *(_BYTE *)v10 )
  {
    v23 = 257;
    v11 = *(_QWORD *)(v10 + 664);
    v12 = *(_QWORD *)(v10 + 672);
    v21[0] = a2;
    v21[1] = v9;
    return sub_921880((unsigned int **)a4, v11, v12, (int)v21, 2, (__int64)v22, 0);
  }
  else
  {
    sub_BCB2B0(*(_QWORD **)(a4 + 72));
    v14 = sub_2463FC0((__int64)a1, a2, (unsigned int **)a4, 0x100u);
    if ( (unsigned __int64)sub_CA1930(&v19) > 0x20 )
    {
      v17 = sub_BCB2B0(*(_QWORD **)(a4 + 72));
      v18 = sub_AD6530(v17, a2);
      return sub_B34240(a4, v14, v18, v9, 0x100u, 0, 0, 0, 0);
    }
    else
    {
      v15 = sub_2463540(a1, a3);
      v16 = (__int64)v15;
      if ( v15 )
        v16 = sub_AD6530((__int64)v15, (__int64)v15);
      return sub_2463EC0((__int64 *)a4, v16, v14, 0x100u, 0);
    }
  }
}
