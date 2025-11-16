// Function: sub_2DB3170
// Address: 0x2db3170
//
__int64 __fastcall sub_2DB3170(__int64 a1, char *a2, unsigned int a3)
{
  size_t v4; // rdx
  __int8 *v5; // rsi
  _BYTE *v7[2]; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v8; // [rsp+10h] [rbp-C0h] BYREF
  __int64 *v9; // [rsp+20h] [rbp-B0h]
  __int64 v10; // [rsp+28h] [rbp-A8h]
  __int64 v11; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v12; // [rsp+40h] [rbp-90h] BYREF
  __int64 v13[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v14[2]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD *v15; // [rsp+70h] [rbp-60h] BYREF
  _QWORD v16[2]; // [rsp+80h] [rbp-50h] BYREF
  __m128i v17; // [rsp+90h] [rbp-40h]

  v4 = 0;
  if ( a2 )
    v4 = strlen(a2);
  sub_B169E0((__int64 *)v7, a2, v4, a3);
  v13[0] = (__int64)v14;
  sub_2DB1EF0(v13, v7[0], (__int64)&v7[0][(unsigned __int64)v7[1]]);
  v15 = v16;
  sub_2DB1EF0((__int64 *)&v15, v9, (__int64)v9 + v10);
  v17 = _mm_loadu_si128(&v12);
  sub_B180C0(a1, (unsigned __int64)v13);
  if ( v15 != v16 )
    j_j___libc_free_0((unsigned __int64)v15);
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0(v13[0]);
  v5 = " cycle";
  if ( a3 != 1 )
    v5 = " cycles";
  sub_B18290(a1, v5, (a3 != 1) + 6LL);
  if ( v9 != &v11 )
    j_j___libc_free_0((unsigned __int64)v9);
  if ( (__int64 *)v7[0] != &v8 )
    j_j___libc_free_0((unsigned __int64)v7[0]);
  return a1;
}
