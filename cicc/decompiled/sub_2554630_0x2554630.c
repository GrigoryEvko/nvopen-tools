// Function: sub_2554630
// Address: 0x2554630
//
__int64 __fastcall sub_2554630(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 **v6; // r15
  unsigned __int8 **v7; // rbx
  __int64 v8; // r12
  unsigned __int8 **v9; // r12
  unsigned __int8 *v10; // rax
  __m128i v11; // rax
  __int64 v12; // r8
  __m128i v14; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int8 *v15; // [rsp+10h] [rbp-40h] BYREF
  char v16; // [rsp+18h] [rbp-38h]

  if ( (unsigned __int8)sub_2509800(a3) == 2 )
    v6 = **(__int64 ****)(*((_QWORD *)sub_250CBE0(a3, a2) + 3) + 16LL);
  else
    v6 = *(__int64 ***)(sub_250D070(a3) + 8);
  v7 = *(unsigned __int8 ***)a4;
  v8 = *(unsigned int *)(a4 + 8);
  v14 = 0;
  v9 = &v7[2 * v8];
  if ( v7 == v9 )
    return sub_ACA8A0(v6);
  while ( 1 )
  {
    v10 = *v7;
    v16 = 1;
    v15 = v10;
    v11.m128i_i64[0] = sub_250C590(&v14, &v15, (__int64)v6);
    v14 = v11;
    if ( v11.m128i_i8[8] )
    {
      v12 = v14.m128i_i64[0];
      if ( !v14.m128i_i64[0] )
        break;
    }
    v7 += 2;
    if ( v9 == v7 )
    {
      if ( !v11.m128i_i8[8] )
        return sub_ACA8A0(v6);
      return v14.m128i_i64[0];
    }
  }
  return v12;
}
