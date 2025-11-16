// Function: sub_1098D90
// Address: 0x1098d90
//
unsigned __int64 __fastcall sub_1098D90(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rcx
  int v4; // edi
  unsigned __int64 v5; // rcx
  unsigned int v6; // eax
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __m128i v11; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v12; // [rsp+10h] [rbp-10h]
  char v13; // [rsp+18h] [rbp-8h]

  v2 = *a1;
  if ( !*a1
    || (_BitScanReverse64(&v3, v2), v4 = 63 - (v3 ^ 0x3F), !a2)
    || (_BitScanReverse64(&v5, a2), v6 = v4 + 63 - (v5 ^ 0x3F), v6 <= 0x3E) )
  {
    v13 = 1;
    return a2 * v2;
  }
  if ( v6 == 63 )
  {
    v7 = a2 * (v2 >> 1);
    if ( v7 >= 0 )
    {
      v8 = 2 * v7;
      if ( (v2 & 1) == 0 )
      {
LABEL_11:
        v12 = v8;
        v13 = 1;
        return v12;
      }
      v9 = a2 + v8;
      if ( a2 < v8 )
        a2 = v8;
      if ( v9 >= a2 )
      {
        v8 = v9;
        goto LABEL_11;
      }
    }
  }
  v11.m128i_i64[1] = 0;
  return _mm_loadu_si128(&v11).m128i_u64[0];
}
