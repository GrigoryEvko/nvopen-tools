// Function: sub_33736A0
// Address: 0x33736a0
//
__int64 __fastcall sub_33736A0(__int64 *a1, unsigned int *a2)
{
  const __m128i *v3; // r12
  __int64 v4; // rdi
  __int64 result; // rax
  __m128i v6; // xmm0
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // rbx
  int v14; // edx
  __int64 v15; // rax
  __int64 *v16; // r8
  __int64 v17; // rsi
  unsigned int v18; // edx
  __m128i v19; // [rsp+0h] [rbp-70h] BYREF
  __int64 v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  int v24; // [rsp+38h] [rbp-38h]

  v3 = (const __m128i *)a1[108];
  v4 = a2[2];
  result = v3[24].m128i_i64[0];
  v6 = _mm_loadu_si128(v3 + 24);
  if ( (_DWORD)v4 )
  {
    if ( *(_DWORD *)(result + 24) != 1 )
    {
      v8 = *(_QWORD *)a2;
      v9 = *(_QWORD *)a2;
      v10 = *(_QWORD *)a2 + 16LL * (unsigned int)(v4 - 1) + 16;
      while ( 1 )
      {
        v11 = *(_QWORD *)(*(_QWORD *)v9 + 40LL);
        if ( *(_QWORD *)v11 == result && *(_DWORD *)(v11 + 8) == v3[24].m128i_i32[2] )
          break;
        v9 += 16;
        if ( v10 == v9 )
        {
          if ( v4 + 1 > (unsigned __int64)a2[3] )
          {
            v19 = v6;
            sub_C8D5F0((__int64)a2, a2 + 4, v4 + 1, 0x10u, v10, v8);
            v8 = *(_QWORD *)a2;
            v4 = a2[2];
            v6 = _mm_load_si128(&v19);
          }
          *(__m128i *)(v8 + 16 * v4) = v6;
          LODWORD(v4) = a2[2] + 1;
          a2[2] = v4;
          v3 = (const __m128i *)a1[108];
          break;
        }
      }
    }
    if ( (_DWORD)v4 == 1 )
    {
      v12 = *(unsigned int *)(*(_QWORD *)a2 + 8LL);
      v13 = **(_QWORD **)a2;
      if ( v13 )
      {
LABEL_9:
        nullsub_1875(v13, v3, 0);
        v22 = v12;
        v21 = v13;
        v3[24].m128i_i64[0] = v13;
        v3[24].m128i_i32[2] = v22;
        sub_33E2B60(v3, 0);
LABEL_10:
        a2[2] = 0;
        return v13;
      }
    }
    else
    {
      v14 = *((_DWORD *)a1 + 212);
      v15 = *a1;
      v23 = 0;
      v16 = &v23;
      v24 = v14;
      if ( v15 )
      {
        if ( &v23 != (__int64 *)(v15 + 48) )
        {
          v17 = *(_QWORD *)(v15 + 48);
          v23 = v17;
          if ( v17 )
          {
            v19.m128i_i64[0] = (__int64)&v23;
            sub_B96E90((__int64)&v23, v17, 1);
            v16 = (__int64 *)v19.m128i_i64[0];
          }
        }
      }
      v19.m128i_i64[0] = (__int64)v16;
      v13 = sub_3402E70(v3, v16, a2);
      v12 = v18;
      if ( v23 )
        sub_B91220(v19.m128i_i64[0], v23);
      v3 = (const __m128i *)a1[108];
      if ( v13 )
        goto LABEL_9;
    }
    v20 = v12;
    v3[24].m128i_i64[0] = 0;
    v3[24].m128i_i32[2] = v20;
    goto LABEL_10;
  }
  return result;
}
