// Function: sub_B24F40
// Address: 0xb24f40
//
char __fastcall sub_B24F40(__m128i *a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 *v8; // rbx
  __int64 v9; // r15
  __int64 i; // r14
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  __int128 v15; // [rsp-10h] [rbp-60h]
  char v16; // [rsp-10h] [rbp-60h]
  __int128 v17; // [rsp+10h] [rbp-40h] BYREF

  v6 = (char *)a2 - (char *)a1;
  v8 = a2;
  *(_QWORD *)&v17 = a4;
  *((_QWORD *)&v17 + 1) = a5;
  if ( (char *)a2 - (char *)a1 > 16 )
  {
    v9 = v6 >> 4;
    for ( i = ((v6 >> 4) - 2) / 2; ; --i )
    {
      LOBYTE(v6) = sub_B24D50((__int64)a1, i, v9, a1[i].m128i_i64[0], a1[i].m128i_i64[1], a6, v17);
      if ( !i )
        break;
    }
  }
  if ( (unsigned __int64)a2 < a3 )
  {
    do
    {
      while ( 1 )
      {
        LOBYTE(v6) = sub_B1DED0((__int64)&v17, v8, a1->m128i_i64);
        if ( (_BYTE)v6 )
          break;
        v8 += 2;
        if ( a3 <= (unsigned __int64)v8 )
          return v6;
      }
      v12 = *v8;
      v15 = v17;
      v8 += 2;
      v13 = *(v8 - 1);
      *((__m128i *)v8 - 1) = _mm_loadu_si128(a1);
      sub_B24D50((__int64)a1, 0, ((char *)a2 - (char *)a1) >> 4, v12, v13, v11, v15);
      LOBYTE(v6) = v16;
    }
    while ( a3 > (unsigned __int64)v8 );
  }
  return v6;
}
