// Function: sub_103B6D0
// Address: 0x103b6d0
//
__int64 __fastcall sub_103B6D0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v3; // rsi
  unsigned __int64 v4; // rbx
  char v5; // al
  __int64 v7; // r15
  __m128i *v9; // rdx
  __m128i *v10; // [rsp+10h] [rbp-90h] BYREF
  __int64 v11; // [rsp+18h] [rbp-88h]
  __m128i v12; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v13[14]; // [rsp+30h] [rbp-70h] BYREF

  v3 = (_QWORD *)(a2 + 48);
  v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v4 == v3 )
    goto LABEL_17;
  if ( !v4 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_17:
    BUG();
  v5 = *(_BYTE *)(v4 - 24);
  v7 = a1 + 16;
  if ( v5 == 31 )
  {
    if ( (*(_DWORD *)(v4 - 20) & 0x7FFFFFF) == 3 )
    {
      *(_QWORD *)a1 = v7;
      *(_QWORD *)(a1 + 8) = 1;
      *(_BYTE *)(a1 + 17) = 0;
      *(_BYTE *)(a1 + 16) = a3 == 0 ? 84 : 70;
      return a1;
    }
    goto LABEL_6;
  }
  if ( v5 != 32 )
  {
LABEL_6:
    *(_QWORD *)a1 = v7;
    sub_103ABA0((__int64 *)a1, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  if ( a3 )
  {
    v13[6] = &v10;
    v12.m128i_i8[0] = 0;
    v13[5] = 0x100000000LL;
    v10 = &v12;
    v13[0] = &unk_49DD210;
    v11 = 0;
    memset(&v13[1], 0, 32);
    sub_CB5980((__int64)v13, 0, 0, 0);
    sub_C49420(*(_QWORD *)(*(_QWORD *)(v4 - 32) + 32LL * (unsigned int)(2 * a3)) + 24LL, (__int64)v13, 1);
    v9 = v10;
    *(_QWORD *)a1 = v7;
    if ( v9 == &v12 )
    {
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v12);
    }
    else
    {
      *(_QWORD *)a1 = v9;
      *(_QWORD *)(a1 + 16) = v12.m128i_i64[0];
    }
    v12.m128i_i8[0] = 0;
    v10 = &v12;
    *(_QWORD *)(a1 + 8) = v11;
    v11 = 0;
    v13[0] = &unk_49DD210;
    sub_CB5840((__int64)v13);
    if ( v10 != &v12 )
      j_j___libc_free_0(v10, v12.m128i_i64[0] + 1);
  }
  else
  {
    *(_QWORD *)a1 = v7;
    *(_WORD *)(a1 + 16) = 25956;
    *(_BYTE *)(a1 + 18) = 102;
    *(_QWORD *)(a1 + 8) = 3;
    *(_BYTE *)(a1 + 19) = 0;
  }
  return a1;
}
