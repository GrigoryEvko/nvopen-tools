// Function: sub_39F7A80
// Address: 0x39f7a80
//
unsigned __int64 __fastcall sub_39F7A80(__m128i *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-1B0h] BYREF
  char v6[296]; // [rsp+10h] [rbp-1A8h] BYREF
  __int64 v7; // [rsp+138h] [rbp-80h]
  __int64 v8; // [rsp+140h] [rbp-78h]
  int v9; // [rsp+150h] [rbp-68h]
  __int64 retaddr; // [rsp+1B8h] [rbp+0h]

  memset(a1, 0, 0xF0u);
  a1[9].m128i_i64[1] = retaddr;
  a1[12].m128i_i64[0] = 0x4000000000000000LL;
  if ( (unsigned int)sub_39F7420(a1, v6) )
    goto LABEL_10;
  if ( (!&_pthread_key_create || pthread_once(&dword_50576F0, sub_39F5C70)) && !byte_5057700[0] )
    memset(byte_5057700, 8, 17);
  if ( byte_5057707 != 8 )
LABEL_10:
    abort();
  v5 = a2;
  if ( (a1[12].m128i_i8[7] & 0x40) != 0 )
    a1[13].m128i_i8[15] = 0;
  v9 = 1;
  a1[3].m128i_i64[1] = (__int64)&v5;
  v8 = 7;
  v7 = 0;
  result = sub_39F6770(a1, (__int64)v6);
  a1[9].m128i_i64[1] = a3;
  return result;
}
