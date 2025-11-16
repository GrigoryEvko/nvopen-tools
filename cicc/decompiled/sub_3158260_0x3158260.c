// Function: sub_3158260
// Address: 0x3158260
//
__int64 __fastcall sub_3158260(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rsi
  unsigned int v7; // [rsp+4h] [rbp-3Ch] BYREF
  char v8[56]; // [rsp+8h] [rbp-38h] BYREF

  result = 0xFFFFFFFFLL;
  v2 = *a1;
  v7 = -1;
  v3 = *(_QWORD *)(v2 + 80);
  v4 = v2 + 72;
  if ( v4 != v3 )
  {
    v5 = 0;
    do
    {
      while ( 1 )
      {
        v6 = v3 - 24;
        if ( !v3 )
          v6 = 0;
        if ( sub_3158140((__int64)a1, v6) )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        ++v5;
        if ( v4 == v3 )
          return v7;
      }
      *(_QWORD *)v8 = v5;
      sub_1098F90(&v7, v8, 8);
      v3 = *(_QWORD *)(v3 + 8);
      ++v5;
    }
    while ( v4 != v3 );
    return v7;
  }
  return result;
}
