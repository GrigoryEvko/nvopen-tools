// Function: sub_F04D90
// Address: 0xf04d90
//
__int64 __fastcall sub_F04D90(__int64 a1, unsigned __int64 a2, __int16 a3, char a4, unsigned int a5)
{
  __int64 v5; // r12
  unsigned __int8 *v7[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+10h] [rbp-20h] BYREF

  sub_F04320((__int64 *)v7, a2, a3, a4, a5);
  v5 = sub_CB6200(a1, v7[0], (size_t)v7[1]);
  if ( (__int64 *)v7[0] != &v8 )
    j_j___libc_free_0(v7[0], v8 + 1);
  return v5;
}
