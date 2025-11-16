// Function: sub_1055A10
// Address: 0x1055a10
//
unsigned __int64 __fastcall sub_1055A10(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rax
  unsigned __int64 v5; // r12
  _QWORD v7[2]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v8; // [rsp+10h] [rbp-70h]
  __int64 v9; // [rsp+18h] [rbp-68h] BYREF
  unsigned int v10; // [rsp+20h] [rbp-60h]
  int v11; // [rsp+58h] [rbp-28h] BYREF
  __int64 v12; // [rsp+60h] [rbp-20h]
  __int64 v13; // [rsp+68h] [rbp-18h]

  v4 = &v9;
  v7[0] = a4;
  v7[1] = 0;
  v8 = 1;
  do
  {
    *v4 = -4096;
    v4 += 2;
  }
  while ( v4 != (__int64 *)&v11 );
  v12 = a2;
  v13 = a3;
  v11 = 0;
  v5 = sub_1054C20((__int64)v7, a1);
  if ( (v8 & 1) == 0 )
    sub_C7D6A0(v9, 16LL * v10, 8);
  return v5;
}
