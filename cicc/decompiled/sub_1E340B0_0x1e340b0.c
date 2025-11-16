// Function: sub_1E340B0
// Address: 0x1e340b0
//
__int64 __fastcall sub_1E340B0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 result; // rax
  unsigned __int64 v6; // r14
  __int64 v7; // r12
  unsigned int v9; // ecx
  unsigned __int8 v10; // [rsp-39h] [rbp-39h]
  unsigned __int64 v11; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v12; // [rsp-30h] [rbp-30h]

  v4 = *a1;
  result = (*a1 >> 2) & 1;
  if ( ((*a1 >> 2) & 1) != 0 )
    return 0;
  v6 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v7 = a1[1] + a2;
    v9 = 8 * sub_15A9520(a4, 0);
    v12 = v9;
    if ( v9 <= 0x40 )
      v11 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & v7;
    else
      sub_16A4EF0((__int64)&v11, v7, 0);
    result = sub_13F8110(v6, 1u, (unsigned __int64)&v11, a4, 0, 0);
    if ( v12 > 0x40 )
    {
      if ( v11 )
      {
        v10 = result;
        j_j___libc_free_0_0(v11);
        return v10;
      }
    }
  }
  return result;
}
