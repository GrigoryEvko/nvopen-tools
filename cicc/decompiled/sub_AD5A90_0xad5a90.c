// Function: sub_AD5A90
// Address: 0xad5a90
//
__int64 __fastcall sub_AD5A90(__int64 a1, _BYTE *a2, unsigned __int8 *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rax
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // [rsp+8h] [rbp-108h]
  __int64 v14; // [rsp+8h] [rbp-108h]
  __int64 v15; // [rsp+8h] [rbp-108h]
  __int64 v16; // [rsp+8h] [rbp-108h]
  _QWORD v17[4]; // [rsp+10h] [rbp-100h] BYREF
  __int64 v18; // [rsp+30h] [rbp-E0h]
  __m128i v19; // [rsp+38h] [rbp-D8h] BYREF
  __m128i v20; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v21; // [rsp+58h] [rbp-B8h]
  __int64 v22; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-A8h]
  __int64 v24; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v25; // [rsp+78h] [rbp-98h]
  char v26; // [rsp+80h] [rbp-90h]
  __int16 v27; // [rsp+90h] [rbp-80h] BYREF
  __m128i v28; // [rsp+98h] [rbp-78h]
  __m128i v29; // [rsp+A8h] [rbp-68h]
  __int64 v30; // [rsp+B8h] [rbp-58h]
  __int64 v31; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v32; // [rsp+C8h] [rbp-48h]
  __int64 v33; // [rsp+D0h] [rbp-40h] BYREF
  unsigned int v34; // [rsp+D8h] [rbp-38h]
  char v35; // [rsp+E0h] [rbp-30h]

  result = sub_AAA5C0(a1, a2, a3);
  if ( !result && a4 != *(_QWORD *)(a1 + 8) )
  {
    v17[2] = a3;
    v17[0] = a1;
    v17[1] = a2;
    v18 = 62;
    v19.m128i_i64[0] = (__int64)v17;
    v19.m128i_i64[1] = 3;
    v20 = 0u;
    v21 = 0;
    v26 = 0;
    v8 = (__int64 *)sub_BD5C60(a1, a2, v7);
    v9 = _mm_loadu_si128(&v19);
    v10 = _mm_loadu_si128(&v20);
    v11 = *v8;
    v35 = 0;
    v28 = v9;
    v27 = v18;
    v12 = v11 + 2120;
    v29 = v10;
    v30 = v21;
    if ( v26 )
    {
      v32 = v23;
      if ( v23 > 0x40 )
        sub_C43780(&v31, &v22);
      else
        v31 = v22;
      v34 = v25;
      if ( v25 > 0x40 )
        sub_C43780(&v33, &v24);
      else
        v33 = v24;
      v35 = 1;
    }
    result = sub_AD4210(v12, *(_QWORD *)(a1 + 8), &v27);
    if ( v35 )
    {
      v35 = 0;
      if ( v34 > 0x40 && v33 )
      {
        v15 = result;
        j_j___libc_free_0_0(v33);
        result = v15;
      }
      if ( v32 > 0x40 && v31 )
      {
        v16 = result;
        j_j___libc_free_0_0(v31);
        result = v16;
      }
    }
    if ( v26 )
    {
      v26 = 0;
      if ( v25 > 0x40 && v24 )
      {
        v13 = result;
        j_j___libc_free_0_0(v24);
        result = v13;
      }
      if ( v23 > 0x40 )
      {
        if ( v22 )
        {
          v14 = result;
          j_j___libc_free_0_0(v22);
          return v14;
        }
      }
    }
  }
  return result;
}
