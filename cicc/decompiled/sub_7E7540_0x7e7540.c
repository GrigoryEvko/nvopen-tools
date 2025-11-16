// Function: sub_7E7540
// Address: 0x7e7540
//
__int64 __fastcall sub_7E7540(const __m128i *a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // r12
  const __m128i *v5; // [rsp+8h] [rbp-48h] BYREF
  __m128i *v6; // [rsp+18h] [rbp-38h] BYREF
  _BYTE v7[48]; // [rsp+20h] [rbp-30h] BYREF

  result = (__int64)&dword_4D03F8C;
  v2 = a1[5].m128i_i64[0];
  v5 = a1;
  if ( !dword_4D03F8C )
    a1[5].m128i_i64[0] = 0;
  if ( v2 )
  {
    v3 = qword_4F06BC0;
    if ( v2 == qword_4F06BC0 )
    {
      result = dword_4D044B4;
      if ( dword_4D044B4 )
      {
        result = a1[4].m128i_i64[1];
        if ( *(_QWORD *)(result + 8) )
          return sub_7E70C0(&v5, &v6, (__int64)v7);
      }
    }
    else
    {
      do
      {
        v4 = v3;
        v3 = *(_QWORD *)(v3 + 32);
      }
      while ( v3 != v2 );
      result = sub_7E71E0(v4, 0, 1);
      if ( (_DWORD)result )
      {
        sub_7E7190(v5, (__int64)v7, &v6);
        sub_7E7530(v4, (__int64)v7);
        return sub_7E2D10((__int64)v5);
      }
    }
  }
  return result;
}
