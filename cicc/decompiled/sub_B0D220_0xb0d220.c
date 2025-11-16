// Function: sub_B0D220
// Address: 0xb0d220
//
__int64 __fastcall sub_B0D220(_QWORD *a1)
{
  unsigned __int64 *v2; // rdx
  unsigned __int64 *v3; // rsi
  __int64 v4; // rdx
  __int64 *v5; // rsi
  __int64 *v6; // rdi
  __int64 v7; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __m128i *v11; // rax
  __int64 v12; // rdx
  _QWORD v13[2]; // [rsp+0h] [rbp-90h] BYREF
  char v14; // [rsp+10h] [rbp-80h]
  __m128i v15; // [rsp+20h] [rbp-70h] BYREF
  __int64 v16; // [rsp+30h] [rbp-60h]
  __int64 *v17; // [rsp+40h] [rbp-50h] BYREF
  __int64 v18; // [rsp+48h] [rbp-48h]
  _BYTE v19[64]; // [rsp+50h] [rbp-40h] BYREF

  v2 = (unsigned __int64 *)a1[3];
  v3 = (unsigned __int64 *)a1[2];
  v17 = (__int64 *)v19;
  v18 = 0x300000000LL;
  sub_AF47B0((__int64)v13, v3, v2);
  if ( v14 )
  {
    v15.m128i_i64[0] = 4096;
    v15.m128i_i64[1] = v13[1];
    v16 = v13[0];
    v9 = (unsigned int)v18;
    v10 = (unsigned int)v18 + 3LL;
    if ( v10 > HIDWORD(v18) )
    {
      sub_C8D5F0(&v17, v19, v10, 8);
      v9 = (unsigned int)v18;
    }
    v11 = (__m128i *)&v17[v9];
    v12 = v16;
    *v11 = _mm_loadu_si128(&v15);
    v11[1].m128i_i64[0] = v12;
    v4 = (unsigned int)(v18 + 3);
    LODWORD(v18) = v18 + 3;
  }
  else
  {
    v4 = (unsigned int)v18;
  }
  v5 = v17;
  v6 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a1[1] & 4) != 0 )
    v6 = (__int64 *)*v6;
  v7 = sub_B0D000(v6, v17, v4, 0, 1);
  if ( v17 != (__int64 *)v19 )
    _libc_free(v17, v5);
  return v7;
}
