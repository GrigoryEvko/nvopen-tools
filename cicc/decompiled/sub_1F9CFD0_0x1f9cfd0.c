// Function: sub_1F9CFD0
// Address: 0x1f9cfd0
//
__int64 *__fastcall sub_1F9CFD0(__int64 **a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // r14
  __m128 v6; // xmm0
  __m128i v7; // xmm1
  unsigned __int64 v8; // rcx
  __int128 v9; // xmm2
  unsigned int v10; // r13d
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // r13
  unsigned __int8 *v16; // rax
  __int64 v17; // rax
  unsigned __int8 v18; // al
  __int64 *v19; // rdi
  int v20; // esi
  __int64 *v21; // rax
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // r9
  int v25; // ecx
  __int128 *v26; // rcx
  __int64 v27; // rsi
  __int64 *v28; // r12
  const void **v29; // r14
  unsigned int v30; // r15d
  __int64 *result; // rax
  __int64 v32; // rdi
  unsigned int v33; // ebx
  __int64 v34; // rsi
  __int64 v35; // [rsp+8h] [rbp-D8h]
  unsigned int v36; // [rsp+10h] [rbp-D0h]
  int v37; // [rsp+14h] [rbp-CCh]
  __int64 v38; // [rsp+18h] [rbp-C8h]
  __int64 v39; // [rsp+30h] [rbp-B0h]
  __int64 v40; // [rsp+38h] [rbp-A8h]
  __int64 (__fastcall *v41)(__int64 *, __int64, __int64, __int64, __int64); // [rsp+40h] [rbp-A0h]
  __int64 v42; // [rsp+48h] [rbp-98h]
  __int64 v43; // [rsp+48h] [rbp-98h]
  __int64 v44; // [rsp+48h] [rbp-98h]
  unsigned __int64 v45; // [rsp+70h] [rbp-70h]
  unsigned int v46; // [rsp+78h] [rbp-68h]
  __int128 *v47; // [rsp+78h] [rbp-68h]
  __int64 *v48; // [rsp+78h] [rbp-68h]
  __int64 v49; // [rsp+80h] [rbp-60h] BYREF
  int v50; // [rsp+88h] [rbp-58h]
  __int64 v51; // [rsp+90h] [rbp-50h] BYREF
  int v52; // [rsp+98h] [rbp-48h]
  char v53; // [rsp+9Ch] [rbp-44h]
  __int64 *v54; // [rsp+A0h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 32);
  v5 = *(_QWORD *)v4;
  v6 = (__m128)_mm_loadu_si128((const __m128i *)(v4 + 40));
  v7 = _mm_loadu_si128((const __m128i *)(v4 + 80));
  v45 = *(_QWORD *)(v4 + 8);
  v8 = *(_QWORD *)v4;
  v9 = (__int128)_mm_loadu_si128((const __m128i *)(v4 + 120));
  v10 = *(_DWORD *)(v4 + 8);
  v35 = *(_QWORD *)(v4 + 80);
  v36 = *(_DWORD *)(v4 + 88);
  v38 = *(_QWORD *)(v4 + 120);
  v37 = *(_DWORD *)(v4 + 128);
  v46 = *(_DWORD *)(*(_QWORD *)(v4 + 160) + 84LL);
  if ( v36 == v37 && *(_QWORD *)(v4 + 120) == *(_QWORD *)(v4 + 80) )
    return (__int64 *)v7.m128i_i64[0];
  v11 = *(_QWORD *)(a2 + 72);
  v49 = v11;
  if ( v11 )
  {
    sub_1623A60((__int64)&v49, v11, 2);
    v8 = v5;
  }
  v12 = (*a1)[6];
  v13 = (*a1)[4];
  v50 = *(_DWORD *)(a2 + 64);
  v14 = v10;
  v15 = a1[1];
  v16 = (unsigned __int8 *)(*(_QWORD *)(v8 + 40) + 16 * v14);
  v42 = v12;
  v39 = *((_QWORD *)v16 + 1);
  v40 = *v16;
  v41 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(*v15 + 264);
  v17 = sub_1E0A0C0(v13);
  v18 = v41(v15, v17, v42, v40, v39);
  v19 = a1[1];
  v20 = v18;
  v21 = *a1;
  v52 = *((_DWORD *)a1 + 4);
  v51 = (__int64)a1;
  v53 = 0;
  v54 = v21;
  v23 = sub_20ACAE0(
          (_DWORD)v19,
          v20,
          v22,
          v5,
          v45,
          v46,
          v6.m128_i64[0],
          v6.m128_i64[1],
          0,
          (__int64)&v51,
          (__int64)&v49);
  if ( v49 )
  {
    v43 = v23;
    sub_161E7C0((__int64)&v49, v49);
    v23 = v43;
  }
  if ( !v23 )
  {
LABEL_20:
    if ( (unsigned __int8)sub_1F9C650(
                            a1,
                            a2,
                            v7.m128i_i64[0],
                            v7.m128i_i64[1],
                            v38,
                            v37,
                            v6,
                            *(double *)v7.m128i_i64,
                            (__m128i)v9) )
      return (__int64 *)a2;
    v34 = *(_QWORD *)(a2 + 72);
    v51 = v34;
    if ( v34 )
      sub_1623A60((__int64)&v51, v34, 2);
    v52 = *(_DWORD *)(a2 + 64);
    result = sub_1F87CB0(
               a1,
               (__int64)&v51,
               v5,
               v45,
               v6.m128_i64[0],
               v6.m128_i64[1],
               (__m128i)v6,
               *(double *)v7.m128i_i64,
               (__m128i)v9,
               v7.m128i_i64[0],
               v7.m128i_i64[1],
               v9,
               v46,
               0);
    if ( v51 )
      goto LABEL_24;
    return result;
  }
  v44 = v23;
  sub_1F81BC0((__int64)a1, v23);
  v25 = *(unsigned __int16 *)(v44 + 24);
  if ( (_WORD)v25 == 32 || v25 == 10 )
  {
    v32 = *(_QWORD *)(v44 + 88);
    v33 = *(_DWORD *)(v32 + 32);
    if ( v33 <= 0x40 )
    {
      if ( *(_QWORD *)(v32 + 24) )
        return (__int64 *)v7.m128i_i64[0];
    }
    else if ( v33 != (unsigned int)sub_16A57B0(v32 + 24) )
    {
      return (__int64 *)v7.m128i_i64[0];
    }
    return (__int64 *)v9;
  }
  if ( (_WORD)v25 == 48 )
    return (__int64 *)v7.m128i_i64[0];
  if ( v25 != 137 )
    goto LABEL_20;
  v26 = *(__int128 **)(v44 + 32);
  v27 = *(_QWORD *)(a2 + 72);
  v28 = *a1;
  v29 = *(const void ***)(*(_QWORD *)(v35 + 40) + 16LL * v36 + 8);
  v30 = *(unsigned __int8 *)(*(_QWORD *)(v35 + 40) + 16LL * v36);
  v51 = v27;
  if ( v27 )
  {
    v47 = v26;
    sub_1623A60((__int64)&v51, v27, 2);
    v26 = v47;
  }
  v52 = *(_DWORD *)(a2 + 64);
  result = sub_1D36A20(
             v28,
             136,
             (__int64)&v51,
             v30,
             v29,
             v24,
             *v26,
             *(__int128 *)((char *)v26 + 40),
             *(_OWORD *)&v7,
             v9,
             v26[5]);
  if ( v51 )
  {
LABEL_24:
    v48 = result;
    sub_161E7C0((__int64)&v51, v51);
    return v48;
  }
  return result;
}
