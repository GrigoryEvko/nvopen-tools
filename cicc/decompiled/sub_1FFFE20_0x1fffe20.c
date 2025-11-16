// Function: sub_1FFFE20
// Address: 0x1fffe20
//
void __fastcall sub_1FFFE20(const __m128i *a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // r8
  __int64 *v9; // r9
  __m128i v10; // xmm0
  const __m128i *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // rax
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // rdi
  __int64 *v19; // rdi
  const __m128i *v20; // [rsp+8h] [rbp-168h]
  char v21; // [rsp+1Fh] [rbp-151h]
  __m128i v22; // [rsp+20h] [rbp-150h] BYREF
  const __m128i *v23; // [rsp+30h] [rbp-140h]
  __int64 *v24; // [rsp+38h] [rbp-138h]
  __int64 v25; // [rsp+40h] [rbp-130h]
  void *v26; // [rsp+50h] [rbp-120h] BYREF
  __int64 v27; // [rsp+58h] [rbp-118h]
  const __m128i *v28; // [rsp+60h] [rbp-110h]
  __m128i v29; // [rsp+68h] [rbp-108h] BYREF
  __int64 (__fastcall *v30)(__m128i *, __m128i *, int); // [rsp+78h] [rbp-F8h]
  _QWORD *(__fastcall *v31)(__int64 *, __int64 *); // [rsp+80h] [rbp-F0h]
  __int64 v32; // [rsp+90h] [rbp-E0h] BYREF
  __int64 *v33; // [rsp+98h] [rbp-D8h]
  __int64 *v34; // [rsp+A0h] [rbp-D0h]
  __int64 v35; // [rsp+A8h] [rbp-C8h]
  int v36; // [rsp+B0h] [rbp-C0h]
  _BYTE v37[184]; // [rsp+B8h] [rbp-B8h] BYREF

  sub_1D236A0((__int64)a1);
  v22.m128i_i64[0] = (__int64)&v32;
  v10 = _mm_loadu_si128(&v22);
  v33 = (__int64 *)v37;
  v11 = (const __m128i *)a1[12].m128i_i64[1];
  v34 = (__int64 *)v37;
  v12 = a1[41].m128i_i64[1];
  v32 = 0;
  v27 = v12;
  a1[41].m128i_i64[1] = (__int64)&v26;
  v35 = 16;
  v28 = a1;
  v26 = &unk_49F9A08;
  v30 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_1FEB650;
  v31 = sub_1FEBB90;
  v13 = a1->m128i_i64[0];
  v36 = 0;
  v22.m128i_i64[0] = v13;
  v14 = a1[1].m128i_i64[0];
  v23 = a1;
  v22.m128i_i64[1] = v14;
  v20 = a1 + 12;
  v24 = &v32;
  v25 = 0;
  v29 = v10;
  do
  {
    if ( v11 == v20 )
      break;
    v21 = 0;
    v11 = v20;
    do
    {
      while ( 1 )
      {
        v16 = v11->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
        v11 = (const __m128i *)v16;
        if ( !v16 )
          BUG();
        v17 = v16 - 8;
        if ( !*(_QWORD *)(v16 + 40) && v17 != a1[11].m128i_i64[0] )
          break;
        v15 = v33;
        if ( v34 != v33 )
          goto LABEL_5;
        v6 = HIDWORD(v35);
        v9 = &v33[HIDWORD(v35)];
        if ( v33 != v9 )
        {
          v19 = 0;
          while ( 1 )
          {
            v6 = *v15;
            if ( v17 == *v15 )
              goto LABEL_6;
            if ( v6 == -2 )
              v19 = v15;
            if ( v9 == ++v15 )
            {
              if ( !v19 )
                break;
              *v19 = v17;
              --v36;
              ++v32;
              goto LABEL_25;
            }
          }
        }
        if ( HIDWORD(v35) < (unsigned int)v35 )
        {
          ++HIDWORD(v35);
          *v9 = v17;
          ++v32;
        }
        else
        {
LABEL_5:
          a2 = v16 - 8;
          sub_16CCBA0((__int64)&v32, v16 - 8);
          if ( !(_BYTE)v6 )
            goto LABEL_6;
        }
LABEL_25:
        a2 = v16 - 8;
        sub_1FFB890(v22.m128i_i64, v16 - 8, v10, a4, a5, v6, v7, v8);
        v21 = 1;
        if ( !*(_QWORD *)(v16 + 40) && v17 != a1[11].m128i_i64[0] )
          break;
LABEL_6:
        if ( a1[12].m128i_i64[1] == v16 )
          goto LABEL_11;
      }
      a2 = v16 - 8;
      v11 = *(const __m128i **)(v16 + 8);
      sub_1D2DE10((__int64)a1, v16 - 8, v6);
    }
    while ( (const __m128i *)a1[12].m128i_i64[1] != v11 );
LABEL_11:
    ;
  }
  while ( v21 );
  sub_1D2D9C0(a1, a2, v6, v7, (__int64)v8, (__int64)v9);
  v26 = &unk_49F9A08;
  if ( v30 )
    v30(&v29, &v29, 3);
  v18 = (unsigned __int64)v34;
  v28[41].m128i_i64[1] = v27;
  if ( (__int64 *)v18 != v33 )
    _libc_free(v18);
}
