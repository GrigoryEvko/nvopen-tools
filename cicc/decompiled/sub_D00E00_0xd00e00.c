// Function: sub_D00E00
// Address: 0xd00e00
//
__m128i *__fastcall sub_D00E00(__m128i *a1, const __m128i *a2, __int64 a3, char a4, char a5)
{
  __int64 v7; // rsi
  __int8 v8; // r15
  unsigned int v9; // edx
  int v11; // eax
  unsigned int v12; // edx
  __int8 v13; // r14
  __m128i v14; // xmm0
  unsigned int v15; // eax
  unsigned int v16; // eax
  bool v17; // cc
  int v19; // eax
  bool v20; // r14
  unsigned int v21; // r14d
  int v22; // eax
  char v23; // [rsp+Ch] [rbp-64h]
  char v24; // [rsp+Ch] [rbp-64h]
  char v25; // [rsp+10h] [rbp-60h]
  unsigned int v26; // [rsp+18h] [rbp-58h]
  unsigned int v27; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-38h]

  v7 = (__int64)&a2[2].m128i_i64[1];
  v8 = *(_BYTE *)(v7 + 17);
  if ( !v8 )
    goto LABEL_7;
  v9 = *(_DWORD *)(a3 + 8);
  if ( v9 > 0x40 )
  {
    v23 = a4;
    v26 = v9;
    v11 = sub_C444A0(a3);
    v12 = v26;
    a4 = v23;
    if ( v11 == v26 - 1 )
    {
      v13 = a2[3].m128i_i8[8];
      if ( !v13 )
        goto LABEL_8;
      v8 = a2[3].m128i_i8[8];
LABEL_20:
      v24 = a4;
      v27 = v12;
      v19 = sub_C444A0(a3);
      a4 = v24;
      v20 = v27 - 1 == v19;
LABEL_21:
      v13 = a4 | v20;
      goto LABEL_8;
    }
    goto LABEL_6;
  }
  if ( *(_QWORD *)a3 != 1 )
  {
LABEL_6:
    v8 = 0;
    if ( a5 )
    {
      v21 = a2[3].m128i_u32[0];
      if ( v21 <= 0x40 )
      {
        v8 = a2[2].m128i_i64[1] == 0;
      }
      else
      {
        v25 = a4;
        v22 = sub_C444A0(v7);
        a4 = v25;
        v8 = v21 == v22;
      }
    }
LABEL_7:
    v13 = a2[3].m128i_i8[8];
    if ( !v13 )
      goto LABEL_8;
    v12 = *(_DWORD *)(a3 + 8);
    if ( v12 > 0x40 )
      goto LABEL_20;
LABEL_27:
    v20 = *(_QWORD *)a3 == 1;
    goto LABEL_21;
  }
  v13 = a2[3].m128i_i8[8];
  if ( v13 )
  {
    v8 = a2[3].m128i_i8[8];
    goto LABEL_27;
  }
LABEL_8:
  sub_C472A0((__int64)&v30, v7, (__int64 *)a3);
  sub_C472A0((__int64)&v28, (__int64)&a2[1].m128i_i64[1], (__int64 *)a3);
  v14 = _mm_loadu_si128(a2);
  a1[1].m128i_i64[0] = a2[1].m128i_i64[0];
  v15 = v29;
  *a1 = v14;
  a1[2].m128i_i32[0] = v15;
  if ( v15 > 0x40 )
    sub_C43780((__int64)&a1[1].m128i_i64[1], (const void **)&v28);
  else
    a1[1].m128i_i64[1] = v28;
  v16 = v31;
  a1[3].m128i_i32[0] = v31;
  if ( v16 > 0x40 )
    sub_C43780((__int64)&a1[2].m128i_i64[1], (const void **)&v30);
  else
    a1[2].m128i_i64[1] = v30;
  v17 = v29 <= 0x40;
  a1[3].m128i_i8[8] = v13;
  a1[3].m128i_i8[9] = v8;
  if ( !v17 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  return a1;
}
