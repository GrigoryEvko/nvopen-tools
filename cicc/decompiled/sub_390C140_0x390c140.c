// Function: sub_390C140
// Address: 0x390c140
//
__m128i *__fastcall sub_390C140(__m128i *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v10; // r8
  __int8 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // rax
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // [rsp-8h] [rbp-100h]
  char v26; // [rsp+1Fh] [rbp-D9h] BYREF
  __int64 v27; // [rsp+20h] [rbp-D8h] BYREF
  __int64 v28; // [rsp+28h] [rbp-D0h] BYREF
  int v29; // [rsp+30h] [rbp-C8h]
  int v30; // [rsp+34h] [rbp-C4h]
  __int64 v31; // [rsp+38h] [rbp-C0h]
  __int64 v32; // [rsp+48h] [rbp-B0h] BYREF
  int v33; // [rsp+50h] [rbp-A8h]
  int v34; // [rsp+54h] [rbp-A4h]
  __int64 v35; // [rsp+58h] [rbp-A0h]
  __m128i v36; // [rsp+68h] [rbp-90h] BYREF
  __m128i v37; // [rsp+78h] [rbp-80h] BYREF
  __int64 v38; // [rsp+88h] [rbp-70h]
  __int64 v39; // [rsp+90h] [rbp-68h]
  __int64 v40; // [rsp+98h] [rbp-60h]
  __int64 v41; // [rsp+A0h] [rbp-58h]
  __int64 v42; // [rsp+A8h] [rbp-50h]
  __int64 v43; // [rsp+B0h] [rbp-48h]
  __int64 v44; // [rsp+B8h] [rbp-40h]
  __int64 v45; // [rsp+C0h] [rbp-38h]

  v36 = 0u;
  v37.m128i_i64[0] = 0;
  v37.m128i_i32[2] = 0;
  v11 = sub_390B240(a2, a3, a5, a4, &v36, &v27, &v26);
  if ( !v11 )
  {
    v12 = a4;
    if ( v36.m128i_i64[0]
      && v36.m128i_i64[1]
      && (v13 = *(_QWORD *)(a2 + 8), v14 = *(__int64 (**)())(*(_QWORD *)v13 + 72LL), v14 != sub_38CB1C0)
      && (v19 = ((__int64 (__fastcall *)(__int64, _QWORD *, __int64, __int64, __int64, __int64 *))v14)(
                  v13,
                  a3,
                  v24,
                  a4,
                  v10,
                  &v27),
          v12 = a4,
          v19) )
    {
      v20 = *(_QWORD *)a5;
      v21 = *(_QWORD *)(a2 + 24);
      v39 = 0;
      v28 = v20;
      LODWORD(v20) = *(_DWORD *)(a5 + 8);
      LODWORD(v41) = 0;
      v29 = v20;
      v30 = *(_DWORD *)(a5 + 12) + 20;
      v31 = *(_QWORD *)(a5 + 16);
      v40 = v37.m128i_i64[0];
      v38 = v36.m128i_i64[0];
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64 *, __int64 *, __int64, _QWORD, __int64, __int64))(*(_QWORD *)v21 + 32LL))(
        v21,
        a2,
        a3,
        a4,
        &v28,
        &v27,
        v36.m128i_i64[0],
        0,
        v37.m128i_i64[0],
        v41);
      v22 = *(_QWORD *)a5;
      LODWORD(v45) = 0;
      v23 = *(_QWORD *)(a2 + 24);
      v44 = 0;
      v32 = v22;
      LODWORD(v22) = *(_DWORD *)(a5 + 8);
      v43 = 0;
      v33 = v22;
      v34 = *(_DWORD *)(a5 + 12) + 24;
      v35 = *(_QWORD *)(a5 + 16);
      v42 = v36.m128i_i64[1];
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64 *, __int64 *, __int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v23 + 32LL))(
        v23,
        a2,
        a3,
        a4,
        &v32,
        &v27,
        v36.m128i_i64[1],
        0,
        0,
        v45);
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, __int64, __int64, __int64 *, __int64, __int64, __int64, __int64))(**(_QWORD **)(a2 + 24) + 32LL))(
        *(_QWORD *)(a2 + 24),
        a2,
        a3,
        v12,
        a5,
        &v27,
        v36.m128i_i64[0],
        v36.m128i_i64[1],
        v37.m128i_i64[0],
        v37.m128i_i64[1]);
    }
  }
  v15 = v27;
  v16 = _mm_loadu_si128(&v36);
  a1->m128i_i8[0] = v11;
  v17 = _mm_loadu_si128(&v37);
  a1->m128i_i64[1] = v15;
  a1[1] = v16;
  a1[2] = v17;
  return a1;
}
