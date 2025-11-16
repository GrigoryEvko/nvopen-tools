// Function: sub_AD5840
// Address: 0xad5840
//
__int64 __fastcall sub_AD5840(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 *v7; // rax
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // [rsp+8h] [rbp-F8h]
  __int64 v13; // [rsp+8h] [rbp-F8h]
  __int64 v14; // [rsp+8h] [rbp-F8h]
  __int64 v15; // [rsp+8h] [rbp-F8h]
  _QWORD v16[2]; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v17; // [rsp+20h] [rbp-E0h]
  __m128i v18; // [rsp+28h] [rbp-D8h] BYREF
  __m128i v19; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v20; // [rsp+48h] [rbp-B8h]
  __int64 v21; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v22; // [rsp+58h] [rbp-A8h]
  __int64 v23; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v24; // [rsp+68h] [rbp-98h]
  char v25; // [rsp+70h] [rbp-90h]
  __int16 v26; // [rsp+80h] [rbp-80h] BYREF
  __m128i v27; // [rsp+88h] [rbp-78h]
  __m128i v28; // [rsp+98h] [rbp-68h]
  __int64 v29; // [rsp+A8h] [rbp-58h]
  __int64 v30; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+B8h] [rbp-48h]
  __int64 v32; // [rsp+C0h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+C8h] [rbp-38h]
  char v34; // [rsp+D0h] [rbp-30h]

  result = sub_AAA0D0((unsigned __int8 *)a1, a2);
  if ( !result )
  {
    v5 = *(_QWORD *)(a1 + 8);
    v6 = *(_QWORD *)(v5 + 24);
    if ( a3 != v6 )
    {
      v16[0] = a1;
      v16[1] = a2;
      v17 = 61;
      v18.m128i_i64[0] = (__int64)v16;
      v18.m128i_i64[1] = 2;
      v19 = 0u;
      v20 = 0;
      v25 = 0;
      v7 = (__int64 *)sub_BD5C60(a1, a2, v5);
      v8 = _mm_loadu_si128(&v18);
      v9 = _mm_loadu_si128(&v19);
      v10 = *v7;
      v34 = 0;
      v27 = v8;
      v26 = v17;
      v11 = v10 + 2120;
      v28 = v9;
      v29 = v20;
      if ( v25 )
      {
        v31 = v22;
        if ( v22 > 0x40 )
          sub_C43780(&v30, &v21);
        else
          v30 = v21;
        v33 = v24;
        if ( v24 > 0x40 )
          sub_C43780(&v32, &v23);
        else
          v32 = v23;
        v34 = 1;
      }
      result = sub_AD4210(v11, v6, &v26);
      if ( v34 )
      {
        v34 = 0;
        if ( v33 > 0x40 && v32 )
        {
          v14 = result;
          j_j___libc_free_0_0(v32);
          result = v14;
        }
        if ( v31 > 0x40 && v30 )
        {
          v15 = result;
          j_j___libc_free_0_0(v30);
          result = v15;
        }
      }
      if ( v25 )
      {
        v25 = 0;
        if ( v24 > 0x40 && v23 )
        {
          v12 = result;
          j_j___libc_free_0_0(v23);
          result = v12;
        }
        if ( v22 > 0x40 )
        {
          if ( v21 )
          {
            v13 = result;
            j_j___libc_free_0_0(v21);
            return v13;
          }
        }
      }
    }
  }
  return result;
}
