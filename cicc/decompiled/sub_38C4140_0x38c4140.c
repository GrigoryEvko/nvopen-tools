// Function: sub_38C4140
// Address: 0x38c4140
//
__int64 __fastcall sub_38C4140(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        const __m128i *a9,
        unsigned int a10)
{
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // esi
  _QWORD v18[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v19[2]; // [rsp+10h] [rbp-50h] BYREF
  __m128i v20; // [rsp+20h] [rbp-40h] BYREF

  v11 = *(_QWORD *)(a2 + 992);
  v18[0] = a5;
  v12 = a2 + 984;
  v19[0] = a3;
  v19[1] = a4;
  v18[1] = a6;
  if ( !v11 )
  {
    v13 = a2 + 984;
LABEL_12:
    v20.m128i_i64[0] = (__int64)&a10;
    LODWORD(v13) = sub_38C3E00((_QWORD *)(a2 + 976), v13, (unsigned int **)&v20);
    goto LABEL_8;
  }
  v13 = a2 + 984;
  do
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v11 + 16);
      v15 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= a10 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v15 )
        goto LABEL_6;
    }
    v13 = v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v14 );
LABEL_6:
  if ( v12 == v13 || a10 < *(_DWORD *)(v13 + 32) )
    goto LABEL_12;
LABEL_8:
  v16 = v13 + 40;
  if ( a9[1].m128i_i8[0] )
    v20 = _mm_loadu_si128(a9);
  sub_38C95C0(a1, v16, (unsigned int)v19, (unsigned int)v18, a8, (unsigned int)&v20, a7);
  return a1;
}
