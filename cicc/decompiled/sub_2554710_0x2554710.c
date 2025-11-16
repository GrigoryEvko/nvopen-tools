// Function: sub_2554710
// Address: 0x2554710
//
__int64 __fastcall sub_2554710(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  bool v7; // zf
  __int64 v8; // rax
  const __m128i *v9; // rbx
  const __m128i *v11; // r10
  _BYTE *v12; // rsi
  unsigned __int8 v13; // al
  char v14; // al
  __int64 v15; // rax
  __m128i v16; // xmm0
  _BYTE *v17; // rax
  unsigned __int64 v18; // r15
  __int64 *v19; // rax
  unsigned int v20; // eax
  int *v21; // r9
  int v22; // eax
  __m128i v23; // [rsp-D8h] [rbp-D8h] BYREF
  const __m128i *v24; // [rsp-C8h] [rbp-C8h]
  __m128i *i; // [rsp-C0h] [rbp-C0h]
  char v26; // [rsp-ADh] [rbp-ADh] BYREF
  unsigned int v27; // [rsp-ACh] [rbp-ACh] BYREF
  _BYTE *v28; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v29; // [rsp-A0h] [rbp-A0h] BYREF
  __int64 v30[4]; // [rsp-98h] [rbp-98h] BYREF
  _BYTE *v31; // [rsp-78h] [rbp-78h] BYREF
  __int64 v32; // [rsp-70h] [rbp-70h]
  _BYTE v33[104]; // [rsp-68h] [rbp-68h] BYREF

  result = 1;
  if ( !*(_QWORD *)(a1 + 360) )
  {
    v7 = *(_BYTE *)(a1 + 105) == 0;
    v31 = v33;
    v32 = 0x300000000LL;
    if ( !v7 )
    {
      v8 = *(unsigned int *)(a1 + 152);
      v9 = *(const __m128i **)(a1 + 144);
      LOBYTE(v29) = 0;
      v11 = (const __m128i *)((char *)v9 + 24 * v8);
      for ( i = (__m128i *)v30; v11 != v9; v9 = (const __m128i *)((char *)v9 + 24) )
      {
        if ( (v9[1].m128i_i8[0] & 1) != 0 )
        {
          v12 = (_BYTE *)v9->m128i_i64[0];
          v13 = *(_BYTE *)v9->m128i_i64[0];
          if ( v13 <= 0x1Cu
            || (v13 & 0xFD) != 0x54
            || (v24 = v11,
                sub_250D230((unsigned __int64 *)i, (unsigned __int64)v12, 1, 0),
                v14 = sub_2526B50(a2, i, a1, (__int64)&v31, 1u, &v29, 1u),
                v11 = v24,
                !v14) )
          {
            v15 = (unsigned int)v32;
            v16 = _mm_loadu_si128(v9);
            if ( (unsigned __int64)(unsigned int)v32 + 1 > HIDWORD(v32) )
            {
              v24 = v11;
              v23 = v16;
              sub_C8D5F0((__int64)&v31, v33, (unsigned int)v32 + 1LL, 0x10u, a5, a6);
              v15 = (unsigned int)v32;
              v16 = _mm_load_si128(&v23);
              v11 = v24;
            }
            *(__m128i *)&v31[16 * v15] = v16;
            LODWORD(v32) = v32 + 1;
          }
        }
      }
      v17 = (_BYTE *)sub_2554630(a2, a1, (__int64 *)(a1 + 72), (__int64)&v31);
      v28 = v17;
      v18 = (unsigned __int64)v17;
      if ( v17 )
      {
        v27 = 1;
        if ( *v17 == 22 )
        {
          if ( !byte_4FEF388 && (unsigned int)sub_2207590((__int64)&byte_4FEF388) )
            sub_2207640((__int64)&byte_4FEF388);
          v19 = (__int64 *)sub_BD5C60(v18);
          v29 = sub_A778C0(v19, 52, 0);
          i = (__m128i *)&v29;
          sub_250D230((unsigned __int64 *)v30, v18, 6, 0);
          v20 = sub_2516380(a2, v30, (__int64)i, 1, 0);
          sub_250C0C0((int *)&v27, v20);
          v21 = (int *)i;
          if ( !byte_4FEF380 )
          {
            v22 = sub_2207590((__int64)&byte_4FEF380);
            v21 = (int *)i;
            if ( v22 )
            {
              sub_2207640((__int64)&byte_4FEF380);
              v21 = (int *)i;
            }
          }
        }
        else
        {
          v21 = (int *)&v29;
        }
        v30[0] = (__int64)&v28;
        v30[1] = a2;
        v30[2] = (__int64)&v27;
        v26 = 0;
        LODWORD(v29) = 1;
        sub_2526370(
          a2,
          (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_256E860,
          (__int64)v30,
          a1,
          v21,
          1,
          &v26,
          1,
          0);
        result = v27;
      }
      else
      {
        result = 1;
      }
      if ( v31 != v33 )
      {
        LODWORD(i) = result;
        _libc_free((unsigned __int64)v31);
        return (unsigned int)i;
      }
    }
  }
  return result;
}
