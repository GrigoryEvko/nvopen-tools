// Function: sub_2ACD960
// Address: 0x2acd960
//
__int64 __fastcall sub_2ACD960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 i; // r14
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 result; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v17; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+18h] [rbp-78h]
  __int64 v19; // [rsp+28h] [rbp-68h] BYREF
  __m128i v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 v21; // [rsp+48h] [rbp-48h] BYREF
  __m128i v22; // [rsp+50h] [rbp-40h] BYREF

  v19 = a6;
  v20.m128i_i64[0] = a4;
  v20.m128i_i64[1] = a5;
  v18 = (a3 - 1) / 2;
  v17 = a3 & 1;
  if ( a2 >= v18 )
  {
    v8 = a1 + 16 * a2;
    if ( (a3 & 1) != 0 )
    {
      v22 = _mm_loadu_si128(&v20);
      goto LABEL_13;
    }
    v7 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v8 = a1 + 32 * (i + 1);
    if ( sub_2ACD630(&v19, v8, v8 - 16) )
    {
      --v7;
      v8 = a1 + 16 * v7;
    }
    v9 = a1 + 16 * i;
    *(_QWORD *)v9 = *(_QWORD *)v8;
    *(_DWORD *)(v9 + 8) = *(_DWORD *)(v8 + 8);
    *(_BYTE *)(v9 + 12) = *(_BYTE *)(v8 + 12);
    if ( v7 >= v18 )
      break;
  }
  if ( !v17 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v7 )
    {
      v14 = v7 + 1;
      v7 = 2 * (v7 + 1) - 1;
      v15 = a1 + 32 * v14 - 16;
      *(_QWORD *)v8 = *(_QWORD *)v15;
      *(_DWORD *)(v8 + 8) = *(_DWORD *)(v15 + 8);
      *(_BYTE *)(v8 + 12) = *(_BYTE *)(v15 + 12);
      v8 = a1 + 16 * v7;
    }
  }
  v10 = _mm_loadu_si128(&v20);
  v21 = v19;
  v22 = v10;
  v11 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v12 = a1 + 16 * v11;
      v8 = a1 + 16 * v7;
      if ( !sub_2ACD630(&v21, v12, (__int64)&v22) )
        break;
      v7 = v11;
      *(_QWORD *)v8 = *(_QWORD *)v12;
      *(_DWORD *)(v8 + 8) = *(_DWORD *)(v12 + 8);
      *(_BYTE *)(v8 + 12) = *(_BYTE *)(v12 + 12);
      if ( a2 >= v11 )
      {
        v8 = a1 + 16 * v11;
        break;
      }
      v11 = (v11 - 1) / 2;
    }
  }
LABEL_13:
  *(_QWORD *)v8 = v22.m128i_i64[0];
  *(_DWORD *)(v8 + 8) = v22.m128i_i32[2];
  result = v22.m128i_u8[12];
  *(_BYTE *)(v8 + 12) = v22.m128i_i8[12];
  return result;
}
