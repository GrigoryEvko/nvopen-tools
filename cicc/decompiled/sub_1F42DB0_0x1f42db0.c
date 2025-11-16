// Function: sub_1F42DB0
// Address: 0x1f42db0
//
__int64 __fastcall sub_1F42DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v7; // r15
  __int64 v8; // r15
  unsigned __int8 v10; // [rsp+Bh] [rbp-75h] BYREF
  unsigned int v11; // [rsp+Ch] [rbp-74h] BYREF
  __int64 v12; // [rsp+10h] [rbp-70h] BYREF
  __int64 v13; // [rsp+18h] [rbp-68h]
  __int64 v14; // [rsp+20h] [rbp-60h] BYREF
  __int64 v15; // [rsp+28h] [rbp-58h]
  __int64 v16; // [rsp+30h] [rbp-50h] BYREF
  __int64 v17; // [rsp+38h] [rbp-48h]
  _BYTE v18[8]; // [rsp+40h] [rbp-40h] BYREF
  __int64 v19; // [rsp+48h] [rbp-38h]
  __int64 v20; // [rsp+50h] [rbp-30h]

  v5 = (unsigned __int8)a4;
  v12 = a4;
  v13 = a5;
  if ( (_BYTE)a4 )
    return *(unsigned __int8 *)(a1 + v5 + 1155);
  if ( (unsigned __int8)sub_1F58D20(&v12) )
  {
    v18[0] = 0;
    v19 = 0;
    LOBYTE(v14) = 0;
    sub_1F426C0(a1, a2, (unsigned int)v12, a5, (__int64)v18, (unsigned int *)&v16, &v14);
    return (unsigned __int8)v14;
  }
  sub_1F40D10((__int64)v18, a1, a2, v12, v13);
  v5 = (unsigned __int8)v19;
  v7 = v20;
  LOBYTE(v14) = v19;
  v15 = v20;
  if ( (_BYTE)v19 )
    return *(unsigned __int8 *)(a1 + v5 + 1155);
  if ( (unsigned __int8)sub_1F58D20(&v14) )
  {
    v18[0] = 0;
    v19 = 0;
    LOBYTE(v11) = 0;
    sub_1F426C0(a1, a2, (unsigned int)v14, v7, (__int64)v18, (unsigned int *)&v16, &v11);
    return (unsigned __int8)v11;
  }
  sub_1F40D10((__int64)v18, a1, a2, v14, v15);
  v5 = (unsigned __int8)v19;
  v8 = v20;
  LOBYTE(v16) = v19;
  v17 = v20;
  if ( (_BYTE)v19 )
    return *(unsigned __int8 *)(a1 + v5 + 1155);
  if ( (unsigned __int8)sub_1F58D20(&v16) )
  {
    v18[0] = 0;
    v19 = 0;
    v10 = 0;
    sub_1F426C0(a1, a2, (unsigned int)v16, v8, (__int64)v18, &v11, &v10);
    return v10;
  }
  else
  {
    sub_1F40D10((__int64)v18, a1, a2, v16, v17);
    return sub_1D5E9F0(a1, a2, (unsigned __int8)v19, v20);
  }
}
