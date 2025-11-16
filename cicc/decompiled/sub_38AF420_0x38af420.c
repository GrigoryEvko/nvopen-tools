// Function: sub_38AF420
// Address: 0x38af420
//
__int64 __fastcall sub_38AF420(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        unsigned __int8 a4,
        unsigned __int8 a5,
        __m128 a6,
        double a7,
        double a8)
{
  __int64 v9; // r15
  __int64 *v10; // rax
  __int64 v11; // rax
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  __m128 *v15; // rax
  _QWORD *v16; // rdi
  unsigned int v17; // r13d
  __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  __int64 v21; // rdi
  const char *v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v27; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+18h] [rbp-B8h] BYREF
  __m128i v29; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v30; // [rsp+30h] [rbp-A0h]
  __m128i v31; // [rsp+40h] [rbp-90h] BYREF
  int v32; // [rsp+50h] [rbp-80h] BYREF
  _QWORD *v33; // [rsp+58h] [rbp-78h]
  int *v34; // [rsp+60h] [rbp-70h]
  int *v35; // [rsp+68h] [rbp-68h]
  __int64 v36; // [rsp+70h] [rbp-60h]
  __int64 v37; // [rsp+78h] [rbp-58h]
  __int64 v38; // [rsp+80h] [rbp-50h]
  __int64 v39; // [rsp+88h] [rbp-48h]
  __int64 v40; // [rsp+90h] [rbp-40h]
  __int64 v41; // [rsp+98h] [rbp-38h]

  if ( *(_DWORD *)(a1 + 64) == 13 )
  {
LABEL_16:
    v17 = a5;
    v19 = a1 + 8;
    LOBYTE(v17) = a4 & a5;
    if ( (a4 & a5) != 0 )
    {
      v20 = *(_QWORD *)(a1 + 56);
      v31.m128i_i64[0] = (__int64)"expected '...' at end of argument list for musttail call in varargs function";
      LOWORD(v32) = 259;
      return (unsigned int)sub_38814C0(v19, v20, (__int64)&v31);
    }
    else
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(v19);
    }
    return v17;
  }
  else
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(a2 + 8) )
      {
        v17 = sub_388AF10(a1, 4, "expected ',' in argument list");
        if ( (_BYTE)v17 )
          return v17;
      }
      if ( *(_DWORD *)(a1 + 64) == 2 )
        break;
      v34 = &v32;
      v9 = *(_QWORD *)(a1 + 56);
      v35 = &v32;
      v27 = 0;
      v31.m128i_i64[0] = 0;
      v32 = 0;
      v33 = 0;
      v36 = 0;
      v37 = 0;
      v38 = 0;
      v39 = 0;
      v40 = 0;
      v41 = 0;
      v29.m128i_i64[0] = (__int64)"expected type";
      LOWORD(v30) = 259;
      if ( (unsigned __int8)sub_3891B00(a1, &v27, (__int64)&v29, 0) )
        goto LABEL_15;
      if ( *(_BYTE *)(v27 + 8) == 8 )
      {
        if ( (unsigned __int8)sub_38A2200((__int64 **)a1, &v28, a3, *(double *)a6.m128_u64, a7, a8) )
          goto LABEL_15;
      }
      else if ( (unsigned __int8)sub_388C730(a1, &v31)
             || (unsigned __int8)sub_38A1070((__int64 **)a1, v27, &v28, a3, *(double *)a6.m128_u64, a7, a8) )
      {
LABEL_15:
        v17 = 1;
        sub_3887AD0(v33);
        return v17;
      }
      v10 = (__int64 *)sub_16498A0(v28);
      v11 = sub_1560BF0(v10, &v31);
      v29.m128i_i64[0] = v9;
      v30 = v11;
      v14 = *(unsigned int *)(a2 + 8);
      v29.m128i_i64[1] = v28;
      if ( (unsigned int)v14 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 24, v12, v13);
        v14 = *(unsigned int *)(a2 + 8);
      }
      a6 = (__m128)_mm_loadu_si128(&v29);
      v15 = (__m128 *)(*(_QWORD *)a2 + 24 * v14);
      *v15 = a6;
      v15[1].m128_u64[0] = v30;
      v16 = v33;
      ++*(_DWORD *)(a2 + 8);
      sub_3887AD0(v16);
      if ( *(_DWORD *)(a1 + 64) == 13 )
        goto LABEL_16;
    }
    v21 = a1 + 8;
    if ( !a4 )
    {
      v31.m128i_i64[0] = (__int64)"unexpected ellipsis in argument list for ";
      v22 = "non-musttail call";
      goto LABEL_21;
    }
    if ( !a5 )
    {
      v31.m128i_i64[0] = (__int64)"unexpected ellipsis in argument list for ";
      v22 = "musttail call in non-varargs function";
LABEL_21:
      v23 = *(_QWORD *)(a1 + 56);
      v31.m128i_i64[1] = (__int64)v22;
      LOWORD(v32) = 771;
      return (unsigned int)sub_38814C0(v21, v23, (__int64)&v31);
    }
    *(_DWORD *)(a1 + 64) = sub_3887100(v21);
    return sub_388AF10(a1, 13, "expected ')' at end of argument list");
  }
}
