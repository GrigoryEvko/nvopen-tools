// Function: sub_1E1AEE0
// Address: 0x1e1aee0
//
void __fastcall sub_1E1AEE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int16 *v3; // r14
  unsigned __int16 v4; // ax
  unsigned __int16 *v5; // r14
  unsigned __int16 i; // ax
  __m128i v7; // [rsp+0h] [rbp-50h] BYREF
  __int64 v8; // [rsp+10h] [rbp-40h]
  __int64 v9; // [rsp+18h] [rbp-38h]
  __int64 v10; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(unsigned __int16 **)(v2 + 32);
  if ( v3 )
  {
    v4 = *v3;
    if ( *v3 )
    {
      do
      {
        ++v3;
        v7.m128i_i64[0] = 805306368;
        v7.m128i_i32[2] = v4;
        v8 = 0;
        v9 = 0;
        v10 = 0;
        sub_1E1A9C0(a1, a2, &v7);
        v4 = *v3;
      }
      while ( *v3 );
      v2 = *(_QWORD *)(a1 + 16);
    }
  }
  v5 = *(unsigned __int16 **)(v2 + 24);
  if ( v5 )
  {
    for ( i = *v5; *v5; i = *v5 )
    {
      ++v5;
      v7.m128i_i64[0] = 0x20000000;
      v7.m128i_i32[2] = i;
      v8 = 0;
      v9 = 0;
      v10 = 0;
      sub_1E1A9C0(a1, a2, &v7);
    }
  }
}
