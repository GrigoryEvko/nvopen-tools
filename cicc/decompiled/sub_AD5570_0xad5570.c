// Function: sub_AD5570
// Address: 0xad5570
//
__int64 __fastcall sub_AD5570(char a1, __int64 a2, unsigned __int8 *a3, char a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 *v10; // rax
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __int64 v13; // r12
  __int64 v14; // r12
  __int64 v15; // [rsp+8h] [rbp-108h]
  __int64 v16; // [rsp+8h] [rbp-108h]
  __int64 v17; // [rsp+8h] [rbp-108h]
  __int64 v18; // [rsp+8h] [rbp-108h]
  _QWORD v19[2]; // [rsp+10h] [rbp-100h] BYREF
  __int16 v20; // [rsp+20h] [rbp-F0h]
  __m128i v21; // [rsp+28h] [rbp-E8h] BYREF
  __m128i v22; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v23; // [rsp+48h] [rbp-C8h]
  __int64 v24; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v25; // [rsp+58h] [rbp-B8h]
  __int64 v26; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v27; // [rsp+68h] [rbp-A8h]
  char v28; // [rsp+70h] [rbp-A0h]
  __int16 v29; // [rsp+80h] [rbp-90h] BYREF
  __m128i v30; // [rsp+88h] [rbp-88h]
  __m128i v31; // [rsp+98h] [rbp-78h]
  __int64 v32; // [rsp+A8h] [rbp-68h]
  __int64 v33; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v34; // [rsp+B8h] [rbp-58h]
  __int64 v35; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v36; // [rsp+C8h] [rbp-48h]
  char v37; // [rsp+D0h] [rbp-40h]

  result = sub_AABE40(a1, (unsigned __int8 *)a2, a3);
  if ( !result && a5 != *(_QWORD *)(a2 + 8) )
  {
    v19[0] = a2;
    v19[1] = a3;
    LOBYTE(v20) = a1;
    HIBYTE(v20) = a4;
    v21.m128i_i64[0] = (__int64)v19;
    v21.m128i_i64[1] = 2;
    v22 = 0u;
    v23 = 0;
    v28 = 0;
    v10 = (__int64 *)sub_BD5C60(a2, a2, v9);
    v11 = _mm_loadu_si128(&v21);
    v12 = _mm_loadu_si128(&v22);
    v13 = *v10;
    v37 = 0;
    v30 = v11;
    v29 = v20;
    v14 = v13 + 2120;
    v31 = v12;
    v32 = v23;
    if ( v28 )
    {
      v34 = v25;
      if ( v25 > 0x40 )
        sub_C43780(&v33, &v24);
      else
        v33 = v24;
      v36 = v27;
      if ( v27 > 0x40 )
        sub_C43780(&v35, &v26);
      else
        v35 = v26;
      v37 = 1;
    }
    result = sub_AD4210(v14, *(_QWORD *)(a2 + 8), &v29);
    if ( v37 )
    {
      v37 = 0;
      if ( v36 > 0x40 && v35 )
      {
        v17 = result;
        j_j___libc_free_0_0(v35);
        result = v17;
      }
      if ( v34 > 0x40 && v33 )
      {
        v18 = result;
        j_j___libc_free_0_0(v33);
        result = v18;
      }
    }
    if ( v28 )
    {
      v28 = 0;
      if ( v27 > 0x40 && v26 )
      {
        v15 = result;
        j_j___libc_free_0_0(v26);
        result = v15;
      }
      if ( v25 > 0x40 )
      {
        if ( v24 )
        {
          v16 = result;
          j_j___libc_free_0_0(v24);
          return v16;
        }
      }
    }
  }
  return result;
}
