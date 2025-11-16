// Function: sub_38E0DD0
// Address: 0x38e0dd0
//
unsigned __int64 __fastcall sub_38E0DD0(_QWORD *a1, unsigned int a2, unsigned int a3, unsigned __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v9; // rbx
  __int64 v10; // rdx
  __int64 (*v11)(); // rax
  __m128i *v12; // rsi
  __int64 v13; // rdi
  __m128i v14; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v15; // [rsp+10h] [rbp-40h]

  result = sub_38DD280((__int64)a1, a4);
  if ( result )
  {
    if ( (a3 & 7) != 0 )
    {
      v13 = a1[1];
      LOWORD(v15) = 259;
      v14.m128i_i64[0] = (__int64)"register save offset is not 8 byte aligned";
      return (unsigned __int64)sub_38BE3D0(v13, a4, (__int64)&v14);
    }
    else
    {
      v9 = (unsigned __int64 *)result;
      v10 = 1;
      v11 = *(__int64 (**)())(*a1 + 16LL);
      if ( v11 != sub_38DBC10 )
        v10 = ((__int64 (__fastcall *)(_QWORD *, unsigned __int64, __int64))v11)(a1, a4, 1);
      v14.m128i_i64[0] = v10;
      v14.m128i_i64[1] = __PAIR64__(a2, a3);
      result = (unsigned int)(a3 > 0x7FFF8) + 4;
      LODWORD(v15) = (a3 > 0x7FFF8) + 4;
      v12 = (__m128i *)v9[10];
      if ( v12 == (__m128i *)v9[11] )
      {
        return sub_38E08D0(v9 + 9, v12, &v14);
      }
      else
      {
        if ( v12 )
        {
          *v12 = _mm_loadu_si128(&v14);
          result = v15;
          v12[1].m128i_i64[0] = v15;
          v12 = (__m128i *)v9[10];
        }
        v9[10] = (unsigned __int64)&v12[1].m128i_u64[1];
      }
    }
  }
  return result;
}
