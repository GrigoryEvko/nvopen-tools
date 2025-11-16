// Function: sub_18B6540
// Address: 0x18b6540
//
_QWORD *__fastcall sub_18B6540(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        unsigned int a6,
        __m128i a7,
        unsigned int *a8)
{
  __int64 v11; // rax
  unsigned int *v12; // r15
  _QWORD *result; // rax
  __int64 **v14; // r15
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __m128i v17; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+10h] [rbp-A0h]
  __int64 v19; // [rsp+18h] [rbp-98h]
  __int64 v20; // [rsp+20h] [rbp-90h] BYREF
  __int16 v21; // [rsp+30h] [rbp-80h]
  _QWORD *v22; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v23[2]; // [rsp+50h] [rbp-60h] BYREF
  int v24; // [rsp+60h] [rbp-50h]
  int v25; // [rsp+74h] [rbp-3Ch]

  v11 = *a1;
  v19 = a2;
  v18 = a3;
  v12 = a8;
  v17 = _mm_loadu_si128(&a7);
  v20 = v11 + 240;
  v21 = 260;
  sub_16E1010((__int64)&v22, (__int64)&v20);
  if ( (unsigned int)(v24 - 31) <= 1 && v25 == 2 )
  {
    if ( v22 != v23 )
      j_j___libc_free_0(v22, v23[0] + 1LL);
    v14 = (__int64 **)a1[6];
    v15 = sub_159C470(a1[7], a6, 0);
    v16 = sub_15A3BA0(v15, v14, 0);
    a7 = _mm_load_si128(&v17);
    return (_QWORD *)sub_18B64A0(a1, v19, v18, a4, a5, v16, (char *)a7.m128i_i64[0], a7.m128i_u64[1]);
  }
  else
  {
    result = v23;
    if ( v22 != v23 )
      result = (_QWORD *)j_j___libc_free_0(v22, v23[0] + 1LL);
    *v12 = a6;
  }
  return result;
}
