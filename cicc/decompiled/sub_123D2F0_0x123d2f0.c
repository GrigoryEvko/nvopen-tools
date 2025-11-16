// Function: sub_123D2F0
// Address: 0x123d2f0
//
__int64 __fastcall sub_123D2F0(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int8 v5; // al
  __m128i *v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // r14
  const __m128i *v10; // r11
  const __m128i *v11; // rcx
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  __m128i *v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rax
  const __m128i *v20; // r12
  const __m128i **v21; // r14
  __int64 *v22; // rcx
  __int64 v23; // r13
  __int64 v24; // rax
  unsigned __int8 v25; // [rsp+17h] [rbp-C9h]
  __int64 v26; // [rsp+20h] [rbp-C0h]
  __int64 v27; // [rsp+28h] [rbp-B8h]
  __int64 v28; // [rsp+30h] [rbp-B0h]
  __int64 *v29; // [rsp+38h] [rbp-A8h]
  __int64 v30; // [rsp+48h] [rbp-98h] BYREF
  __m128i v31; // [rsp+50h] [rbp-90h] BYREF
  __m128i v32; // [rsp+60h] [rbp-80h] BYREF
  __int64 v33; // [rsp+70h] [rbp-70h]
  unsigned __int64 v34; // [rsp+78h] [rbp-68h]
  __int64 v35; // [rsp+80h] [rbp-60h] BYREF
  int v36; // [rsp+88h] [rbp-58h] BYREF
  _QWORD *v37; // [rsp+90h] [rbp-50h]
  int *v38; // [rsp+98h] [rbp-48h]
  int *v39; // [rsp+A0h] [rbp-40h]
  __int64 v40; // [rsp+A8h] [rbp-38h]

  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  else
  {
    v36 = 0;
    v37 = 0;
    v38 = &v36;
    v39 = &v36;
    v40 = 0;
    while ( 1 )
    {
      v5 = sub_1239130(a1, v32.m128i_i64, &v35, (__int64)(a3[1] - *a3) >> 4);
      if ( v5 )
      {
        v25 = v5;
        goto LABEL_42;
      }
      v6 = (__m128i *)a3[1];
      if ( v6 == (__m128i *)a3[2] )
      {
        sub_D78DC0((__int64)a3, v6, &v32);
      }
      else
      {
        if ( v6 )
        {
          *v6 = _mm_loadu_si128(&v32);
          v6 = (__m128i *)a3[1];
        }
        a3[1] = v6 + 1;
      }
      if ( *(_DWORD *)(a1 + 240) != 4 )
        break;
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    }
    v7 = 13;
    v25 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( !v25 )
    {
      v28 = a1 + 1656;
      if ( v38 != &v36 )
      {
        v26 = a1;
        v9 = (__int64)v38;
        do
        {
          v32.m128i_i32[0] = *(_DWORD *)(v9 + 32);
          v10 = *(const __m128i **)(v9 + 48);
          v11 = *(const __m128i **)(v9 + 40);
          v32.m128i_i64[1] = 0;
          v33 = 0;
          v34 = 0;
          v12 = (char *)v10 - (char *)v11;
          if ( v10 == v11 )
          {
            v13 = 0;
          }
          else
          {
            if ( v12 > 0x7FFFFFFFFFFFFFF0LL )
              sub_4261EA(a1, v7, v8);
            v13 = sub_22077B0((char *)v10 - (char *)v11);
            v10 = *(const __m128i **)(v9 + 48);
            v11 = *(const __m128i **)(v9 + 40);
          }
          v32.m128i_i64[1] = v13;
          v33 = v13;
          v34 = v13 + v12;
          if ( v10 == v11 )
          {
            v15 = v13;
          }
          else
          {
            v14 = (__m128i *)v13;
            v15 = v13 + (char *)v10 - (char *)v11;
            do
            {
              if ( v14 )
                *v14 = _mm_loadu_si128(v11);
              ++v14;
              ++v11;
            }
            while ( (__m128i *)v15 != v14 );
          }
          v33 = v15;
          v16 = *(_QWORD *)(v26 + 1664);
          if ( v16 )
          {
            v17 = v28;
            do
            {
              while ( 1 )
              {
                v7 = *(_QWORD *)(v16 + 16);
                v18 = *(_QWORD *)(v16 + 24);
                if ( *(_DWORD *)(v16 + 32) >= v32.m128i_i32[0] )
                  break;
                v16 = *(_QWORD *)(v16 + 24);
                if ( !v18 )
                  goto LABEL_28;
              }
              v17 = v16;
              v16 = *(_QWORD *)(v16 + 16);
            }
            while ( v7 );
LABEL_28:
            if ( v17 != v28 && v32.m128i_i32[0] >= *(_DWORD *)(v17 + 32) )
              goto LABEL_31;
          }
          else
          {
            v17 = v28;
          }
          v7 = v17;
          v31.m128i_i64[0] = (__int64)&v32;
          v19 = sub_123CD60((_QWORD *)(v26 + 1648), v17, (unsigned int **)&v31);
          v15 = v33;
          v17 = v19;
          v13 = v32.m128i_i64[1];
LABEL_31:
          if ( v13 != v15 )
          {
            v27 = v9;
            v20 = (const __m128i *)v13;
            v21 = (const __m128i **)(v17 + 40);
            v22 = &v31.m128i_i64[1];
            v23 = v17;
            do
            {
              while ( 1 )
              {
                v24 = *a3 + 16LL * v20->m128i_u32[0];
                v31 = _mm_loadu_si128(v20);
                v7 = *(_QWORD *)(v23 + 48);
                v30 = v24;
                if ( v7 != *(_QWORD *)(v23 + 56) )
                  break;
                ++v20;
                v29 = v22;
                sub_12149F0(v21, (const __m128i *)v7, &v30, v22);
                v22 = v29;
                if ( (const __m128i *)v15 == v20 )
                  goto LABEL_38;
              }
              if ( v7 )
              {
                *(_QWORD *)v7 = v24;
                *(_QWORD *)(v7 + 8) = v31.m128i_i64[1];
                v7 = *(_QWORD *)(v23 + 48);
              }
              v7 += 16;
              ++v20;
              *(_QWORD *)(v23 + 48) = v7;
            }
            while ( (const __m128i *)v15 != v20 );
LABEL_38:
            v9 = v27;
            v15 = v32.m128i_i64[1];
          }
          if ( v15 )
          {
            v7 = v34 - v15;
            j_j___libc_free_0(v15, v34 - v15);
          }
          a1 = v9;
          v9 = sub_220EEE0(v9);
        }
        while ( (int *)v9 != &v36 );
      }
    }
LABEL_42:
    sub_1207E40(v37);
  }
  return v25;
}
