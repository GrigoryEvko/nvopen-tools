// Function: sub_32441A0
// Address: 0x32441a0
//
__int64 __fastcall sub_32441A0(__int64 a1, int a2, __int8 a3)
{
  unsigned __int64 *v3; // rdi
  __m128i *v5; // r8
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // r12d
  __int8 v10; // al
  int v11; // edx
  __int8 v12[4]; // [rsp+8h] [rbp-18h] BYREF
  int v13[3]; // [rsp+Ch] [rbp-14h] BYREF

  v3 = *(unsigned __int64 **)(a1 + 16);
  v13[0] = a2;
  v12[0] = a3;
  v5 = (__m128i *)v3[96];
  v6 = v3[95];
  v7 = (__int64)((__int64)v5->m128i_i64 - v6) >> 4;
  if ( (_DWORD)v7 )
  {
    v8 = 0;
    while ( *(_DWORD *)v6 != v13[0] || *(_BYTE *)(v6 + 4) != a3 )
    {
      ++v8;
      v6 += 16LL;
      if ( (_DWORD)v7 == v8 )
        goto LABEL_8;
    }
    return v8;
  }
  v8 = 0;
LABEL_8:
  if ( v5 == (__m128i *)v3[97] )
  {
    sub_3244010(v3 + 95, v5, v13, v12);
    return v8;
  }
  if ( v5 )
  {
    v10 = v12[0];
    v11 = v13[0];
    v5->m128i_i64[1] = 0;
    v5->m128i_i32[0] = v11;
    v5->m128i_i8[4] = v10;
    v5 = (__m128i *)v3[96];
  }
  v3[96] = (unsigned __int64)&v5[1];
  return v8;
}
