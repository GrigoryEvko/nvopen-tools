// Function: sub_7FC1E0
// Address: 0x7fc1e0
//
void *__fastcall sub_7FC1E0(
        const __m128i *a1,
        __int64 a2,
        __m128i *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8)
{
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  const __m128i *v14; // rax
  __m128i *v15; // rbx
  _QWORD *v16; // rax
  __int64 v17; // r15
  const __m128i *v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // rax
  _QWORD *v21; // r14
  __int64 *v22; // rax
  void *v23; // r13
  __int64 v25; // rsi
  __m128i *v26; // rax
  __m128i *v27; // rax
  __int64 v28; // r14
  __int64 v29; // rdx
  _QWORD *v30; // rax
  __int64 v31; // rcx
  _QWORD *v32; // r8
  _QWORD *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // r13
  const __m128i *v36; // rax
  _QWORD *v37; // rbx
  __int64 v38; // rax
  __int64 v39; // r13
  const __m128i *v40; // rax
  _QWORD *v41; // r13
  _QWORD *v42; // r14
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  _BYTE *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rsi
  _QWORD *v50; // rax
  __int64 v51; // rcx
  _QWORD *v52; // r13
  _QWORD *v53; // [rsp+0h] [rbp-80h]
  __int64 v54; // [rsp+8h] [rbp-78h]
  bool v55; // [rsp+17h] [rbp-69h]
  _QWORD *v56; // [rsp+18h] [rbp-68h]
  _QWORD *v59; // [rsp+30h] [rbp-50h]
  __int64 v60; // [rsp+30h] [rbp-50h]
  _QWORD *v61; // [rsp+30h] [rbp-50h]
  _QWORD *v62; // [rsp+38h] [rbp-48h]
  _BYTE v63[4]; // [rsp+44h] [rbp-3Ch] BYREF
  const __m128i *v64[7]; // [rsp+48h] [rbp-38h] BYREF

  sub_72BA30(byte_4F06A51[0]);
  v62 = sub_7F5ED0(a2);
  v10 = a6 | a7;
  if ( !a1 || (v56 = 0, v10) )
  {
    v11 = sub_8D46C0(a2);
    v12 = sub_7F5F50(v11, a6);
    v56 = sub_73A830(v12, byte_4F06A51[0]);
    v62[2] = v56;
  }
  if ( !a8 )
  {
    v13 = qword_4F18B38;
    if ( qword_4F18B38 )
      goto LABEL_6;
LABEL_24:
    v28 = sub_7E1C10();
    if ( HIDWORD(qword_4F0688C) )
      v29 = sub_7E1C10();
    else
      v29 = sub_72CBE0();
    v60 = v29;
    v30 = sub_7259C0(7);
    v31 = v30[21];
    v32 = v30;
    v30[20] = v60;
    *(_BYTE *)(v31 + 16) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v31 + 16) & 0xFD;
    if ( v28 )
    {
      v61 = v30;
      v33 = sub_724EF0(v28);
      v32 = v61;
      *(_QWORD *)v61[21] = v33;
    }
    qword_4F18B38 = sub_72D2E0(v32);
    v13 = qword_4F18B38;
    v14 = (const __m128i *)sub_724DC0();
    v64[0] = v14;
    if ( a4 )
      goto LABEL_7;
    goto LABEL_29;
  }
  v26 = (__m128i *)sub_8D46C0(a2);
  v27 = sub_7FB1A0(v26, 1u, 0, 1, a4);
  v13 = qword_4F18B38;
  a4 = (__int64)v27;
  if ( !qword_4F18B38 )
    goto LABEL_24;
LABEL_6:
  v14 = (const __m128i *)sub_724DC0();
  v64[0] = v14;
  if ( a4 )
  {
LABEL_7:
    v59 = sub_731330(a4);
    goto LABEL_8;
  }
LABEL_29:
  v34 = (__int64)v14;
  sub_72BB40(v13, v14);
  v59 = sub_73A720(v64[0], v34);
LABEL_8:
  sub_724E30((__int64)v64);
  if ( !v10 )
  {
    v15 = a3;
    if ( a1 )
    {
      a1[1].m128i_i64[0] = (__int64)a3;
      v15 = (__m128i *)a1;
    }
    a3[1].m128i_i64[0] = (__int64)v62;
    v16 = v56;
    if ( !v56 )
      v16 = v62;
    v16[2] = v59;
    v17 = sub_7F9D60();
    v18 = (const __m128i *)sub_724DC0();
    v64[0] = v18;
    if ( a5 )
    {
      v19 = sub_731330(a5);
    }
    else
    {
      v25 = (__int64)v18;
      sub_72BB40(v17, v18);
      v19 = sub_73A720(v64[0], v25);
    }
    sub_724E30((__int64)v64);
    v59[2] = v19;
    if ( a1 )
    {
      v20 = sub_72CBE0();
      v21 = sub_7F89D0("__cxa_vec_ctor", &qword_4F18B68, v20, v15);
      v22 = sub_7E8090(a1, 1u);
      v23 = sub_73DF90((__int64)v21, v22);
    }
    else
    {
      v44 = sub_7E1C10();
      v23 = sub_7F89D0("__cxa_vec_new", &qword_4F18B80, v44, v15);
    }
    if ( a4 )
      sub_8255D0(a4);
    if ( a5 )
      sub_8255D0(a5);
    return v23;
  }
  v35 = sub_7F9D60();
  v36 = (const __m128i *)sub_724DC0();
  v64[0] = v36;
  if ( a5 )
  {
    v37 = sub_731330(a5);
  }
  else
  {
    v49 = (__int64)v36;
    sub_72BB40(v35, v36);
    v37 = sub_73A720(v64[0], v49);
  }
  sub_724E30((__int64)v64);
  v55 = 0;
  if ( a7 )
  {
    v55 = (unsigned int)sub_6013A0(a7, (__int64)v63) != 0;
    if ( !a6 )
    {
      v38 = sub_7D3810(3u);
      a6 = *(_QWORD *)(sub_87AC70(v38, v64) + 88);
    }
  }
  v39 = qword_4F18B20;
  if ( !qword_4F18B20 )
  {
    v53 = sub_72BA30(byte_4F06A51[0]);
    v54 = sub_7E1C10();
    v50 = sub_7259C0(7);
    v51 = v50[21];
    v52 = v50;
    v50[20] = v54;
    *(_BYTE *)(v51 + 16) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v51 + 16) & 0xFD;
    if ( v53 )
      *(_QWORD *)v50[21] = sub_724EF0((__int64)v53);
    qword_4F18B20 = sub_72D2E0(v52);
    v39 = qword_4F18B20;
  }
  v40 = (const __m128i *)sub_724DC0();
  v64[0] = v40;
  if ( a6 )
  {
    v41 = sub_731330(a6);
  }
  else
  {
    v48 = (__int64)v40;
    sub_72BB40(v39, v40);
    v41 = sub_73A720(v64[0], v48);
  }
  sub_724E30((__int64)v64);
  v42 = sub_7F7FC0(a7);
  a3[1].m128i_i64[0] = (__int64)v62;
  v56[2] = v59;
  if ( v55 )
  {
    v45 = sub_7F52E0();
    v46 = sub_73E110((__int64)v42, v45);
    v59[2] = v37;
    v37[2] = v41;
    v41[2] = v46;
    v47 = sub_7E1C10();
    v23 = sub_7F89D0("__cxa_vec_new3", &qword_4F18B70, v47, a3);
    if ( !a4 )
      goto LABEL_41;
    goto LABEL_40;
  }
  v59[2] = v37;
  v37[2] = v41;
  v41[2] = v42;
  v43 = sub_7E1C10();
  v23 = sub_7F89D0("__cxa_vec_new2", &qword_4F18B78, v43, a3);
  if ( a4 )
LABEL_40:
    sub_8255D0(a4);
LABEL_41:
  if ( a5 )
    sub_8255D0(a5);
  if ( a6 )
    sub_8255D0(a6);
  if ( a7 )
    sub_8255D0(a7);
  return v23;
}
