// Function: sub_13F3340
// Address: 0x13f3340
//
__int64 __fastcall sub_13F3340(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // r10
  __int64 result; // rax
  __int64 v16; // [rsp+8h] [rbp-78h]
  unsigned int v18; // [rsp+18h] [rbp-68h]
  unsigned int v19; // [rsp+18h] [rbp-68h]
  int v20; // [rsp+20h] [rbp-60h] BYREF
  __int64 v21; // [rsp+28h] [rbp-58h]
  unsigned int v22; // [rsp+30h] [rbp-50h]
  __int64 v23; // [rsp+38h] [rbp-48h]
  unsigned int v24; // [rsp+40h] [rbp-40h]

  v9 = sub_157EB90(a5);
  v10 = sub_1632FA0(v9);
  v11 = sub_13E7A30(a1 + 4, *a1, v10, a1[3]);
  v20 = 0;
  v16 = v11;
  v12 = sub_13EFC20(v11, a3, a5, a6, &v20, a7);
  v13 = a4;
  if ( !v12 )
  {
    sub_13EFEC0(v16);
    sub_13EFC20(v16, a3, a5, a6, &v20, a7);
    v13 = a4;
  }
  result = sub_13E9B70(a2, v13, &v20, v10, a1[2]);
  if ( v20 == 3 )
  {
    if ( v24 > 0x40 && v23 )
    {
      v18 = result;
      j_j___libc_free_0_0(v23);
      result = v18;
    }
    if ( v22 > 0x40 )
    {
      if ( v21 )
      {
        v19 = result;
        j_j___libc_free_0_0(v21);
        return v19;
      }
    }
  }
  return result;
}
