// Function: sub_1D5E9F0
// Address: 0x1d5e9f0
//
__int64 __fastcall sub_1D5E9F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // r13d
  __int64 v6; // [rsp+0h] [rbp-60h] BYREF
  __int64 v7; // [rsp+8h] [rbp-58h]
  unsigned __int8 v8; // [rsp+1Bh] [rbp-45h] BYREF
  char v9; // [rsp+1Ch] [rbp-44h] BYREF
  _BYTE v10[8]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v11; // [rsp+28h] [rbp-38h]
  __int64 v12; // [rsp+30h] [rbp-30h]

  v6 = a3;
  v7 = a4;
  if ( (_BYTE)a3 )
    return *(unsigned __int8 *)(a1 + (unsigned __int8)a3 + 1155);
  v5 = a4;
  if ( (unsigned __int8)sub_1F58D20(&v6) )
  {
    v10[0] = 0;
    v11 = 0;
    v8 = 0;
    sub_1F426C0(a1, a2, v6, v5, (unsigned int)v10, (unsigned int)&v9, (__int64)&v8);
    return v8;
  }
  else
  {
    sub_1F40D10(v10, a1, a2, v6, v7);
    return sub_1D5E9F0(a1, a2, (unsigned __int8)v11, v12);
  }
}
