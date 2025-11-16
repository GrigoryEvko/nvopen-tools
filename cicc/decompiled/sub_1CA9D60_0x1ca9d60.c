// Function: sub_1CA9D60
// Address: 0x1ca9d60
//
__int64 __fastcall sub_1CA9D60(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  int v5; // r12d
  unsigned __int8 *v6; // r9
  __int64 v7; // r13
  unsigned __int8 v8; // bl
  __int64 v9; // rsi
  char v10; // al
  unsigned __int8 v12; // [rsp+7h] [rbp-99h]
  __int64 v13; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v14; // [rsp+18h] [rbp-88h]
  unsigned __int8 v15; // [rsp+2Fh] [rbp-71h] BYREF
  __int64 v16; // [rsp+30h] [rbp-70h] BYREF
  __int64 v17; // [rsp+38h] [rbp-68h]
  __int64 v18; // [rsp+40h] [rbp-60h]
  __int64 v19; // [rsp+50h] [rbp-50h] BYREF
  __int64 v20; // [rsp+58h] [rbp-48h]
  __int64 v21; // [rsp+60h] [rbp-40h]
  int v22; // [rsp+68h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v15 = 0;
  v19 = 0;
  v20 = 0;
  if ( v4 )
    v4 -= 24;
  v21 = 0;
  v5 = 0;
  v22 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  sub_191E690((__int64)&v16, v4);
  v12 = 0;
  v6 = &v15;
  while ( 1 )
  {
    v7 = v17;
    ++v5;
    v13 = v16;
    if ( v16 == v17 )
      break;
    v8 = 0;
    do
    {
      v9 = *(_QWORD *)(v7 - 8);
      v14 = v6;
      v7 -= 8;
      v10 = sub_1CA8CD0(a1, v9, v5, a3, (__int64)&v19, v6);
      v6 = v14;
      v8 |= v10;
    }
    while ( v13 != v7 );
    if ( !v8 )
    {
      v7 = v16;
      break;
    }
    v12 = v15;
    if ( !v15 )
    {
      v12 = v8;
      v7 = v16;
      break;
    }
  }
  if ( v7 )
    j_j___libc_free_0(v7, v18 - v7);
  j___libc_free_0(v20);
  return v12;
}
