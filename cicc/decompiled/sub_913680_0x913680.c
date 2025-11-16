// Function: sub_913680
// Address: 0x913680
//
__int64 __fastcall sub_913680(_QWORD *a1, const __m128i *a2, __int64 a3, unsigned __int32 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // r13
  _QWORD *v8; // rcx
  const __m128i *v9; // rax
  __int64 result; // rax
  __m128i *v11; // rsi
  const __m128i **v12; // rax
  const __m128i **v13; // r12
  bool v14; // dl
  bool v15; // cl
  __m128i *v16; // rsi
  _BYTE *v17; // rsi
  const __m128i *v20; // [rsp+18h] [rbp-68h]
  const __m128i **v21; // [rsp+28h] [rbp-58h] BYREF
  __m128i v22; // [rsp+30h] [rbp-50h] BYREF
  const __m128i *v23; // [rsp+40h] [rbp-40h]

  v4 = a1[54];
  v5 = (a1[55] - v4) >> 3;
  if ( !(_DWORD)v5 )
  {
LABEL_15:
    v22 = (__m128i)4uLL;
    v23 = a2;
    if ( &a2[256] != 0 && a2 != 0 && a2 != (const __m128i *)-8192LL )
      sub_BD73F0(&v22);
    v12 = (const __m128i **)sub_22077B0(48);
    v13 = v12;
    if ( v12 )
    {
      *v12 = (const __m128i *)4;
      v12[1] = 0;
      v12[2] = 0;
      v12[3] = 0;
      v12[4] = 0;
      v12[5] = 0;
      v14 = &v23[256] != 0;
      v15 = &v23[512] != 0;
      if ( !v23 || (v12[2] = v23, !v14) || !v15 )
      {
        v21 = v12;
LABEL_27:
        v22.m128i_i64[0] = a3;
        result = a4;
        v22.m128i_i32[2] = a4;
        v16 = (__m128i *)v13[4];
        if ( v16 == v13[5] )
        {
          result = sub_913500(v13 + 3, v16, &v22);
        }
        else
        {
          if ( v16 )
          {
            *v16 = _mm_loadu_si128(&v22);
            v16 = (__m128i *)v13[4];
          }
          v13[4] = v16 + 1;
        }
        v17 = (_BYTE *)a1[55];
        if ( v17 == (_BYTE *)a1[56] )
          return (__int64)sub_90AAB0((__int64)(a1 + 54), v17, &v21);
        if ( v17 )
        {
          result = (__int64)v21;
          *(_QWORD *)v17 = v21;
          v17 = (_BYTE *)a1[55];
        }
        a1[55] = v17 + 8;
        return result;
      }
      sub_BD6050(v12, v22.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
    }
    v21 = v13;
    if ( &v23[256] != 0 && &v23[512] != 0 && v23 )
    {
      sub_BD60C0(&v22);
      v13 = v21;
    }
    goto LABEL_27;
  }
  v6 = 0;
  v7 = 8LL * (unsigned int)(v5 - 1);
  while ( 1 )
  {
    v8 = *(_QWORD **)(v4 + v6);
    v22 = (__m128i)4uLL;
    v23 = (const __m128i *)v8[2];
    v9 = v23;
    if ( &v23[256] != 0 && v23 != 0 && v23 != (const __m128i *)-8192LL )
    {
      sub_BD6050(&v22, *v8 & 0xFFFFFFFFFFFFFFF8LL);
      v9 = v23;
      if ( v23 != 0 && &v23[256] != 0 && v23 != (const __m128i *)-8192LL )
      {
        v20 = v23;
        sub_BD60C0(&v22);
        v9 = v20;
      }
    }
    if ( a2 == v9 )
      break;
    if ( v7 == v6 )
      goto LABEL_15;
    v4 = a1[54];
    v6 += 8;
  }
  result = *(_QWORD *)(a1[54] + v6);
  v22.m128i_i32[2] = a4;
  v22.m128i_i64[0] = a3;
  v11 = *(__m128i **)(result + 32);
  if ( v11 == *(__m128i **)(result + 40) )
    return sub_913500((const __m128i **)(result + 24), v11, &v22);
  if ( v11 )
  {
    *v11 = _mm_loadu_si128(&v22);
    v11 = *(__m128i **)(result + 32);
  }
  *(_QWORD *)(result + 32) = v11 + 1;
  return result;
}
