// Function: sub_702840
// Address: 0x702840
//
__int64 *__fastcall sub_702840(char *src, __int64 a2, __int64 a3, _DWORD *a4, int a5, int a6, __m128i *a7, __int64 *a8)
{
  int v9; // r12d
  __int64 v11; // rax
  size_t v12; // rax
  int v13; // r15d
  __int64 v14; // rsi
  __m128i *v15; // rdi
  __int64 v18; // [rsp+8h] [rbp-3E8h] BYREF
  int v19; // [rsp+14h] [rbp-3DCh] BYREF
  _QWORD *v20; // [rsp+18h] [rbp-3D8h] BYREF
  unsigned __int64 v21; // [rsp+20h] [rbp-3D0h]
  __int64 v22; // [rsp+28h] [rbp-3C8h] BYREF
  __m128i v23; // [rsp+30h] [rbp-3C0h]
  __m128i v24; // [rsp+40h] [rbp-3B0h]
  __m128i v25; // [rsp+50h] [rbp-3A0h]
  _BYTE v26[160]; // [rsp+60h] [rbp-390h] BYREF
  __m128i v27[22]; // [rsp+100h] [rbp-2F0h] BYREF
  _QWORD v28[50]; // [rsp+260h] [rbp-190h] BYREF

  v9 = a2;
  v18 = a3;
  v11 = *(_QWORD *)a4;
  v21 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v23 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v24 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v25 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v22 = v11;
  v12 = strlen(src);
  sub_878540(src, v12);
  v13 = sub_87FF90(v21, &v22);
  sub_6E1E00(4u, (__int64)v26, 0, 0);
  v14 = a2 != 0;
  if ( (unsigned int)sub_84C4B0(
                       v13,
                       v14,
                       v9,
                       0,
                       0,
                       (unsigned int)&v18,
                       1,
                       1,
                       1,
                       0,
                       0,
                       1,
                       a6,
                       0,
                       (__int64)a4,
                       a5,
                       0,
                       (__int64)&v19,
                       (__int64)v27,
                       (__int64)&v20) )
  {
    v14 = (__int64)v28;
    v15 = v27;
    sub_7022F0(v27, v28, v20, 1, 0, 0, v19, 0, (__int64 *)&dword_4F077C8, a4, &dword_4F077C8, (__int64)a7, 0, a8);
  }
  else
  {
    v15 = a7;
    sub_6E6260(a7);
    if ( a8 )
      *a8 = 0;
  }
  return sub_6E2B30((__int64)v15, v14);
}
