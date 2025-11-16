// Function: sub_B8D9F0
// Address: 0xb8d9f0
//
__int64 __fastcall sub_B8D9F0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // r13
  __int64 *j; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rsi
  __int64 v12; // r12
  __int64 v14; // [rsp+8h] [rbp-D8h]
  __int64 v15; // [rsp+18h] [rbp-C8h]
  __int64 *i; // [rsp+28h] [rbp-B8h]
  __m128i v17; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v18; // [rsp+40h] [rbp-A0h]
  __int64 *v19[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v20; // [rsp+60h] [rbp-80h]
  _BYTE *v21; // [rsp+70h] [rbp-70h] BYREF
  __int64 v22; // [rsp+78h] [rbp-68h]
  _BYTE v23[96]; // [rsp+80h] [rbp-60h] BYREF

  v21 = v23;
  v22 = 0x600000000LL;
  sub_B8D880(&v17, (__int64)a2);
  sub_B8D590(v19, a2);
  v4 = v18;
  for ( i = v20; i != v18; v4 = v18 )
  {
    if ( (unsigned __int8)sub_B8D8F0((__int64)a3, (const void *)*v4, v4[1]) )
    {
      v5 = sub_B8D530(a1, *v4, v4[1], v4[2], v4[3]);
      v6 = (unsigned int)v22;
      if ( (unsigned __int64)(unsigned int)v22 + 1 > HIDWORD(v22) )
      {
        v14 = v5;
        sub_C8D5F0(&v21, v23, (unsigned int)v22 + 1LL, 8);
        v6 = (unsigned int)v22;
        v5 = v14;
      }
      *(_QWORD *)&v21[8 * v6] = v5;
      LODWORD(v22) = v22 + 1;
    }
    v18 += 4;
    sub_B8D830((__int64)&v17);
  }
  sub_B8D880(&v17, (__int64)a3);
  sub_B8D590(v19, a3);
  v7 = v20;
  for ( j = v18; v7 != v18; j = v18 )
  {
    if ( (unsigned __int8)sub_B8D8F0((__int64)a2, (const void *)*j, j[1]) )
    {
      v9 = sub_B8D530(a1, *j, j[1], j[2], j[3]);
      v10 = (unsigned int)v22;
      if ( (unsigned __int64)(unsigned int)v22 + 1 > HIDWORD(v22) )
      {
        v15 = v9;
        sub_C8D5F0(&v21, v23, (unsigned int)v22 + 1LL, 8);
        v10 = (unsigned int)v22;
        v9 = v15;
      }
      *(_QWORD *)&v21[8 * v10] = v9;
      LODWORD(v22) = v22 + 1;
    }
    v18 += 4;
    sub_B8D830((__int64)&v17);
  }
  v11 = v21;
  v12 = sub_B9C770(a1, v21, (unsigned int)v22, 0, 1);
  if ( v21 != v23 )
    _libc_free(v21, v11);
  return v12;
}
