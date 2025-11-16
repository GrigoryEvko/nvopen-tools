// Function: sub_2728DA0
// Address: 0x2728da0
//
void __fastcall sub_2728DA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // ebx
  unsigned __int64 v10; // r15
  __int64 v11; // rcx
  bool v12; // bl
  unsigned __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+8h] [rbp-38h]

  sub_D19730((__int64)&v13, a2, a1, a4, a5, a6);
  v9 = v14;
  if ( v14 )
  {
    v10 = v13;
    if ( v14 <= 0x40 )
    {
      v11 = 64 - v14;
      v12 = v13 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14);
    }
    else
    {
      v12 = v9 == (unsigned int)sub_C445E0((__int64)&v13);
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    if ( !v12 )
      sub_27289C0(a1, a2, v6, v11, v7, v8);
  }
}
