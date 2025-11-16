// Function: sub_9C57B0
// Address: 0x9c57b0
//
__int64 __fastcall sub_9C57B0(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  __int64 *v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 *v8; // r14
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __int64 v12; // [rsp+0h] [rbp-70h]
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v15; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h]
  _QWORD v17[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = a2;
  v5 = a1;
  v6 = *a3;
  v7 = a3[1];
  v15 = v17;
  v8 = &a1[v3];
  if ( &a1[v3] && !a1 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v14 = (v3 * 8) >> 3;
  v9 = v17;
  if ( (unsigned __int64)v3 > 15 )
  {
    v12 = v7;
    v13 = v6;
    v9 = (_BYTE *)sub_22409D0(&v15, &v14, 0);
    v7 = v12;
    v15 = v9;
    v6 = v13;
    v17[0] = v14;
  }
  if ( a1 != v8 )
  {
    do
    {
      v10 = *v5++;
      *v9++ = v10;
    }
    while ( v8 != v5 );
    v9 = v15;
  }
  v16 = v14;
  v9[v14] = 0;
  sub_2241130(a3, v6 + v7 - *a3, 0, v15, v16);
  if ( v15 != v17 )
    j_j___libc_free_0(v15, v17[0] + 1LL);
  return 0;
}
