// Function: sub_637960
// Address: 0x637960
//
__int64 __fastcall sub_637960(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 result; // rax
  unsigned int v6; // edx
  bool v7; // zf
  __int64 v8; // [rsp+8h] [rbp-68h] BYREF
  __m128i v9; // [rsp+10h] [rbp-60h] BYREF
  __int128 v10; // [rsp+30h] [rbp-40h]

  memset(&v9, 0, 32);
  v10 = 0;
  v8 = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE10(v10) |= 1u;
  DWORD2(v10) |= 0x8020009u;
  if ( a3 && *(_BYTE *)(a3 + 8) != 2 )
    BYTE11(v10) |= 0x10u;
  if ( dword_4D04964 )
    BYTE8(v10) |= 0x80u;
  else
    BYTE9(v10) |= 1u;
  result = (__int64)sub_637180(a2, a4, &v9, 0, dword_4D048B8, 0, &v8);
  if ( (*(_BYTE *)(a1 + 193) & 2) != 0 )
    result = sub_630970(a1, (__int64)&v9, (__int64)&v8);
  if ( a3 )
  {
    result = v9.m128i_i64[1];
    v6 = dword_4D048B8;
    *(_BYTE *)(v9.m128i_i64[1] + 49) |= 0x20u;
    *(_QWORD *)(a3 + 24) = result;
    *(_BYTE *)(a3 + 9) |= 4u;
    if ( !v6 )
    {
      v7 = *(_BYTE *)(result + 48) == 6;
      *(_QWORD *)(result + 16) = 0;
      if ( v7 && (*(_BYTE *)(*(_QWORD *)(result + 56) + 192LL) & 1) == 0 )
        *(_BYTE *)(result + 48) = 2;
    }
  }
  return result;
}
