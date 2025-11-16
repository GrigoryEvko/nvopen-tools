// Function: sub_8A6360
// Address: 0x8a6360
//
__m128i *__fastcall sub_8A6360(
        __int64 a1,
        __m128i *a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __int64 *a6,
        int a7,
        int *a8,
        __m128i *a9)
{
  __int64 v10; // rax
  int v11; // r13d
  __m128i *v12; // r14
  __m128i *result; // rax
  __int64 v14; // rbx
  __int64 v15; // rdx
  int v16; // eax
  __m128i *v17; // [rsp+0h] [rbp-60h]
  int v19; // [rsp+14h] [rbp-4Ch]

  v10 = a5[2];
  if ( !(_DWORD)v10 )
    return sub_8A55D0(a1, a2, a3, a4, 0, 0, a6, a7, a8, a9);
  v11 = v10 - 1;
  v19 = v10 - 1;
  if ( (int)v10 - 1 < 0 )
    return 0;
  v12 = a2;
  result = 0;
  v14 = 24LL * v11;
  while ( !*a8 )
  {
    v16 = a7;
    v15 = v14 + *a5;
    BYTE1(v16) = BYTE1(a7) | 0x40;
    if ( !v11 )
      v16 = a7;
    if ( (*(_BYTE *)(v15 + 16) & 4) != 0 )
      v16 |= 0x2000u;
    if ( (*(_BYTE *)(v15 + 16) & 8) != 0 )
      v16 |= 0x80000u;
    result = sub_8A55D0(a1, v12, a3, a4, *(_QWORD *)(v15 + 8), *(_QWORD *)v15, a6, v16, a8, a9);
    if ( v19 > v11 )
    {
      v17 = result;
      sub_725130(v12->m128i_i64);
      result = v17;
    }
    --v11;
    v14 -= 24;
    if ( v11 == -1 )
      break;
    v12 = result;
  }
  return result;
}
