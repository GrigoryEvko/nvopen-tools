// Function: sub_29AA260
// Address: 0x29aa260
//
__int64 *__fastcall sub_29AA260(__int64 a1, unsigned int a2, __int64 *a3, __int64 a4, char a5)
{
  unsigned int v5; // ebx
  __int64 *result; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  _QWORD *v13; // rax
  _QWORD *v14; // r15
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-C0h]
  __int64 v17; // [rsp+8h] [rbp-B8h]
  __int64 *v19; // [rsp+18h] [rbp-A8h]
  __int64 *i; // [rsp+48h] [rbp-78h]
  __int64 v21[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v22[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v23; // [rsp+80h] [rbp-40h]

  result = &a3[a4];
  v19 = result;
  for ( i = a3; v19 != i; result = i )
  {
    v8 = *i;
    v22[0] = *(_QWORD *)(*i + 8);
    v9 = sub_B6E160(**(__int64 ***)a1, a2, (__int64)v22, 1);
    v10 = *(__int64 **)(a1 + 8);
    v23 = 257;
    v11 = *v10;
    v21[1] = v8;
    v12 = 0;
    v21[0] = v11;
    if ( v9 )
      v12 = *(_QWORD *)(v9 + 24);
    v13 = sub_BD2CC0(88, 3u);
    if ( v13 )
    {
      v14 = v13;
      v5 = v5 & 0xE0000000 | 3;
      sub_B44260((__int64)v13, **(_QWORD **)(v12 + 16), 56, v5, 0, 0);
      v14[9] = 0;
      sub_B4A290((__int64)v14, v12, v9, v21, 2, (__int64)v22, 0, 0);
    }
    else
    {
      v14 = 0;
    }
    if ( a5 )
    {
      v7 = v16;
      LOWORD(v7) = 0;
      v16 = v7;
      sub_B44220(v14, **(_QWORD **)(a1 + 16) + 24LL, v7);
    }
    else
    {
      v15 = v17;
      LOWORD(v15) = 0;
      v17 = v15;
      sub_B44220(v14, **(_QWORD **)(a1 + 24) + 24LL, v15);
    }
    ++i;
  }
  return result;
}
