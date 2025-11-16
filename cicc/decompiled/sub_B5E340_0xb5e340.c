// Function: sub_B5E340
// Address: 0xb5e340
//
__int64 __fastcall sub_B5E340(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r12
  __int64 v5; // rbx
  const char *v6; // r15
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v10; // [rsp+0h] [rbp-40h]
  size_t n; // [rsp+8h] [rbp-38h]

  v3 = a2 - a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 > 0 )
  {
    v6 = *(const char **)a3;
    n = *(_QWORD *)(a3 + 8);
    v10 = qword_4F818F0;
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = v4 + 8 * (v5 >> 1);
        if ( strncmp((const char *)(v10 + *(unsigned int *)(v8 + 4)), v6, n) >= 0 )
          break;
        v4 = v8 + 8;
        v5 = v5 - v7 - 1;
        if ( v5 <= 0 )
          return v4;
      }
      v5 >>= 1;
    }
    while ( v7 > 0 );
  }
  return v4;
}
