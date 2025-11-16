// Function: sub_20E6B90
// Address: 0x20e6b90
//
__int64 __fastcall sub_20E6B90(__int64 *a1, unsigned int *a2)
{
  __int64 v3; // rax
  __int64 *v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v3 = sub_20E6AF0((__int64)a1, a2);
  v5 = (__int64)v4;
  v6 = v3;
  v11 = a1[5];
  if ( v3 == a1[3] && v4 == a1 + 1 )
  {
    sub_20E5700(a1[2]);
    a1[2] = 0;
    a1[3] = v5;
    a1[4] = v5;
    a1[5] = 0;
  }
  else if ( (__int64 *)v3 == v4 )
  {
    return 0;
  }
  else
  {
    do
    {
      v7 = v6;
      v6 = sub_220EF30(v6);
      v8 = sub_220F330(v7, a1 + 1);
      j_j___libc_free_0(v8, 48);
      v9 = a1[5] - 1;
      a1[5] = v9;
    }
    while ( v5 != v6 );
    v11 -= v9;
  }
  return v11;
}
