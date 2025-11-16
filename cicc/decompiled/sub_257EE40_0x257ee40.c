// Function: sub_257EE40
// Address: 0x257ee40
//
__int64 __fastcall sub_257EE40(__int64 a1, unsigned __int64 a2)
{
  __m128i v2; // rax
  __int64 v3; // r13
  __int64 v4; // r14
  unsigned int v5; // r12d
  __int64 v6; // rax
  __m128i v8; // [rsp+0h] [rbp-30h] BYREF

  v2.m128i_i64[0] = sub_250D2C0(a2, **(_QWORD **)a1);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD *)(a1 + 16);
  v8 = v2;
  v5 = sub_2554560(v3, &v8, 40, 0);
  if ( !(_BYTE)v5 )
  {
    if ( v4 )
    {
      v6 = sub_257D970(v3, v8.m128i_i64[0], v8.m128i_i64[1], v4, 0, 0, 1);
      if ( v6 )
        return *(unsigned __int8 *)(v6 + 97);
    }
  }
  return v5;
}
