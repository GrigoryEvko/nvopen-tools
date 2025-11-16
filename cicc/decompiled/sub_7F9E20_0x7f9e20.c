// Function: sub_7F9E20
// Address: 0x7f9e20
//
_QWORD *__fastcall sub_7F9E20(const __m128i *a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r13
  __int64 v7; // rax
  __int64 *v8; // r15
  const __m128i *v9; // rbx
  __int64 v10; // rax
  __m128i *v11; // rax
  __m128i *v12; // r15
  __int64 v13; // rax
  _BYTE *v14; // rax
  _BYTE *v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rsi
  _QWORD *v19; // r15
  __int64 v20; // rdi
  const __m128i *v21; // r14
  _QWORD *v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 *v27; // rdi
  __int64 v28; // rsi
  _BYTE *v29; // rax
  __int64 v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rax
  void *v35; // r13
  __int64 v36; // rax
  _BYTE *v37; // r13
  _QWORD *v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  const __m128i *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // [rsp+0h] [rbp-60h]
  int v53; // [rsp+Ch] [rbp-54h]
  __int64 v54; // [rsp+10h] [rbp-50h]
  const __m128i *v55; // [rsp+18h] [rbp-48h]
  _QWORD *v56; // [rsp+18h] [rbp-48h]
  _QWORD *v57; // [rsp+18h] [rbp-48h]
  _DWORD v58[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = (__int64 *)a1;
  v55 = a2;
  sub_72BA30(byte_4F06A51[0]);
  v7 = a1->m128i_i64[0];
  v58[0] = 0;
  v53 = 0;
  v54 = v7;
  if ( a4 )
    v53 = sub_6013A0(a4, (__int64)v58);
  v8 = (__int64 *)a1;
  v9 = (const __m128i *)sub_7F5ED0(v54);
  if ( v58[0] )
    v8 = sub_7E8090(a1, 1u);
  v10 = sub_7E1C10();
  v11 = (__m128i *)sub_73E110((__int64)v8, v10);
  v11[1].m128i_i64[0] = (__int64)v9;
  v12 = v11;
  v13 = sub_7F9D60();
  v14 = sub_73E110(a3, v13);
  v15 = v14;
  if ( a2 )
  {
    v9[1].m128i_i64[0] = (__int64)v14;
    if ( a4 )
      goto LABEL_7;
    v12[1].m128i_i64[0] = (__int64)a2;
    a2[1].m128i_i64[0] = (__int64)v9;
    v41 = sub_72CBE0();
    v18 = &qword_4F18B40;
    v19 = sub_7F89D0("__cxa_vec_dtor", &qword_4F18B40, v41, v12);
LABEL_16:
    if ( v15[24] == 20 )
      goto LABEL_11;
    goto LABEL_17;
  }
  v48 = sub_8D46C0(v54);
  v49 = sub_7F5F50(v48, 0);
  v50 = sub_73A830(v49, byte_4F06A51[0]);
  v9[1].m128i_i64[0] = (__int64)v50;
  v50[2] = v15;
  if ( !a4 )
  {
    v51 = sub_72CBE0();
    v18 = &qword_4F18B58;
    v19 = sub_7F89D0("__cxa_vec_delete", &qword_4F18B58, v51, v12);
    goto LABEL_16;
  }
LABEL_7:
  v16 = sub_7F7FC0(a4);
  if ( v58[0] )
  {
    v45 = (const __m128i *)sub_7F6050(a1);
    v12[1].m128i_i64[0] = (__int64)v45;
    v45[1].m128i_i64[0] = (__int64)v9;
    v9[1].m128i_i64[0] = (__int64)v15;
    v55 = v45;
    v46 = sub_72CBE0();
    a1[1].m128i_i64[0] = (__int64)sub_7F89D0("__cxa_vec_dtor", &qword_4F18B40, v46, v12);
    v47 = sub_72CBE0();
    v18 = (__int64 *)1;
    v19 = sub_73DBF0(0x5Bu, v47, (__int64)a1);
    v5 = sub_7E8090(a1, 1u);
  }
  else if ( v53 )
  {
    v52 = (__int64)v16;
    v43 = sub_7F52E0();
    *((_QWORD *)v15 + 2) = sub_73E110(v52, v43);
    v44 = sub_72CBE0();
    v18 = &qword_4F18B48;
    v19 = sub_7F89D0("__cxa_vec_delete3", &qword_4F18B48, v44, v12);
  }
  else
  {
    *((_QWORD *)v15 + 2) = v16;
    v17 = sub_72CBE0();
    v18 = &qword_4F18B50;
    v19 = sub_7F89D0("__cxa_vec_delete2", &qword_4F18B50, v17, v12);
  }
  sub_8255D0(a4);
  if ( v15[24] == 20 )
  {
LABEL_11:
    sub_8255D0(*((_QWORD *)v15 + 7));
    if ( !v58[0] )
      return v19;
    goto LABEL_12;
  }
LABEL_17:
  if ( !v58[0] )
    return v19;
LABEL_12:
  v20 = a4;
  v21 = (const __m128i *)sub_7F6130(v54);
  v22 = sub_731330(a4);
  v26 = sub_7E1C30(v20, v18, v23, v24, v25);
  v27 = v5;
  v28 = v26;
  v29 = sub_73E130(v5, v26);
  *((_QWORD *)v29 + 2) = v21;
  v30 = (__int64)v29;
  v34 = sub_7E1C30(v27, v28, v31, v32, v33);
  v35 = sub_73DBF0(0x33u, v34, v30);
  v36 = sub_7E1C10();
  v37 = sub_73E110((__int64)v35, v36);
  if ( v53 )
  {
    v56 = sub_7E8090(v55, 1u);
    v56[2] = sub_73B8B0(v9, 0);
    v57 = sub_73DBF0(0x29u, v9->m128i_i64[0], (__int64)v56);
    v57[2] = sub_73B8B0(v21, 0);
    v38 = sub_73DBF0(0x27u, v9->m128i_i64[0], (__int64)v57);
    v38[2] = sub_73B8B0(v21, 0);
    *((_QWORD *)v37 + 2) = v38;
  }
  else
  {
    *((_QWORD *)v37 + 2) = sub_73B8B0(v21, 0);
  }
  v22[2] = v37;
  v39 = sub_72CBE0();
  v19[2] = sub_73DBF0(0x69u, v39, (__int64)v22);
  v40 = sub_72CBE0();
  return sub_73DBF0(0x5Bu, v40, (__int64)v19);
}
