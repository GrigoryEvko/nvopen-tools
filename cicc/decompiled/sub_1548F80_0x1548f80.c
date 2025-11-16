// Function: sub_1548F80
// Address: 0x1548f80
//
__int64 __fastcall sub_1548F80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v8; // r12
  __int64 i; // r14
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // r8
  __int64 v13; // r12
  __int64 v14; // r14
  __int64 v16; // r10
  __int64 v17; // rax
  __int64 v20; // [rsp+18h] [rbp-68h]
  __m128i v22; // [rsp+30h] [rbp-50h] BYREF
  __int64 v23; // [rsp+40h] [rbp-40h]

  v8 = (a3 - 1) / 2;
  v20 = a3 & 1;
  if ( a2 >= v8 )
  {
    v11 = a1 + 16 * a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v10 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1);
    v11 = a1 + 32 * (i + 1);
    if ( sub_1548C70((__int64 *)&a7, *(_QWORD *)v11, *(_QWORD *)(v11 - 16)) )
    {
      --v10;
      v11 = a1 + 16 * v10;
    }
    v12 = a1 + 16 * i;
    *(_QWORD *)v12 = *(_QWORD *)v11;
    *(_DWORD *)(v12 + 8) = *(_DWORD *)(v11 + 8);
    if ( v10 >= v8 )
      break;
  }
  if ( !v20 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v10 )
    {
      v16 = v10 + 1;
      v10 = 2 * (v10 + 1) - 1;
      v17 = a1 + 32 * v16 - 16;
      *(_QWORD *)v11 = *(_QWORD *)v17;
      *(_DWORD *)(v11 + 8) = *(_DWORD *)(v17 + 8);
      v11 = a1 + 16 * v10;
    }
  }
  v23 = a8;
  v22 = _mm_loadu_si128((const __m128i *)&a7);
  v13 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      v14 = a1 + 16 * v13;
      v11 = a1 + 16 * v10;
      if ( !sub_1548C70(v22.m128i_i64, *(_QWORD *)v14, a4) )
        break;
      v10 = v13;
      *(_QWORD *)v11 = *(_QWORD *)v14;
      *(_DWORD *)(v11 + 8) = *(_DWORD *)(v14 + 8);
      if ( a2 >= v13 )
      {
        v11 = a1 + 16 * v13;
        break;
      }
      v13 = (v13 - 1) / 2;
    }
  }
LABEL_13:
  *(_QWORD *)v11 = a4;
  *(_DWORD *)(v11 + 8) = a5;
  return a5;
}
