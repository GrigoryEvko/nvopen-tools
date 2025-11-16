// Function: sub_7FB1A0
// Address: 0x7fb1a0
//
__m128i *__fastcall sub_7FB1A0(__m128i *a1, unsigned int a2, int a3, int a4, __int64 a5)
{
  const __m128i *i; // r12
  __int64 v6; // rbx
  __m128i *v7; // r13
  __int64 v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // r14
  __m128i *v11; // r12
  __int8 v12; // al
  __int64 v13; // r14
  __int64 *v14; // r15
  __m128i *v15; // rax
  __m128i *v16; // r14
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rsi
  __m128i *v21; // rbx
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  _QWORD *v27; // rdx
  const __m128i *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // r9
  __m128i *v32; // r14
  __int64 *v33; // rax
  _BYTE *v34; // rdi
  __int64 v36; // rax
  __m128i *v37; // rax
  _QWORD *v38; // r9
  __int64 v39; // rdx
  _QWORD *v40; // rax
  _BYTE *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  _QWORD *v46; // r8
  __m128i *v47; // rax
  _QWORD *v48; // r8
  _QWORD *v49; // r9
  __m128i *v50; // rax
  _QWORD **v51; // r15
  __int64 v52; // rax
  __int64 v53; // [rsp+0h] [rbp-160h]
  _QWORD *v54; // [rsp+0h] [rbp-160h]
  _BYTE *v55; // [rsp+10h] [rbp-150h]
  _QWORD *v56; // [rsp+10h] [rbp-150h]
  __int64 v57; // [rsp+10h] [rbp-150h]
  _QWORD *v58; // [rsp+10h] [rbp-150h]
  _QWORD *v59; // [rsp+10h] [rbp-150h]
  _QWORD *v60; // [rsp+10h] [rbp-150h]
  __m128i *v62; // [rsp+20h] [rbp-140h]
  _QWORD *v63; // [rsp+28h] [rbp-138h]
  __m128i *v64; // [rsp+28h] [rbp-138h]
  unsigned int v68; // [rsp+4Ch] [rbp-114h] BYREF
  _BYTE v69[32]; // [rsp+50h] [rbp-110h] BYREF
  _BYTE v70[240]; // [rsp+70h] [rbp-F0h] BYREF

  for ( i = a1; a1[8].m128i_i8[12] == 12; a1 = (__m128i *)a1[10].m128i_i64[0] )
    ;
  v6 = sub_72D2E0(a1);
  v7 = sub_73C570(i, 1);
  v63 = sub_72BA30(byte_4F06A51[0]);
  v8 = sub_72CBE0();
  v9 = sub_7259C0(7);
  v9[20] = v8;
  v10 = v9;
  *(_BYTE *)(v9[21] + 16LL) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v9[21] + 16LL) & 0xFD;
  if ( v6 )
    *(_QWORD *)v9[21] = sub_724EF0(v6);
  v11 = sub_725FD0();
  v11[10].m128i_i8[12] = 2;
  v12 = v11[5].m128i_i8[8];
  v11[12].m128i_i8[1] |= 0x10u;
  v11[9].m128i_i64[1] = (__int64)v10;
  v11[5].m128i_i8[8] = v12 & 0x8F | 0x10;
  sub_7362F0((__int64)v11, 0);
  v13 = *(_QWORD *)(v11[9].m128i_i64[1] + 168);
  if ( a3 )
  {
    v46 = sub_724EF0((__int64)v63);
    **(_QWORD **)v13 = v46;
    if ( a4 )
    {
      v58 = v46;
      v14 = sub_7F54F0((__int64)v11, 1, 0, &v68);
      sub_7F6C60((__int64)v14, v68, (__int64)v70);
      v47 = sub_7E2270(*(_QWORD *)(*(_QWORD *)v13 + 8LL));
      v48 = v58;
      v49 = 0;
    }
    else
    {
      v54 = v46;
      v51 = *(_QWORD ***)v13;
      v52 = sub_72D2E0(v7);
      v60 = sub_724EF0(v52);
      **v51 = v60;
      v14 = sub_7F54F0((__int64)v11, 1, 0, &v68);
      sub_7F6C60((__int64)v14, v68, (__int64)v70);
      v47 = sub_7E2270(*(_QWORD *)(*(_QWORD *)v13 + 8LL));
      v48 = v54;
      v49 = v60;
    }
    v62 = v47;
    v14[5] = (__int64)v47;
    v59 = v49;
    v50 = sub_7E2270(v48[1]);
    v38 = v59;
    v53 = (__int64)v50;
    *(_QWORD *)(v14[5] + 112) = v50;
    v39 = v14[5] + 112;
    if ( a4 )
      goto LABEL_8;
  }
  else
  {
    if ( a4 )
    {
      v14 = sub_7F54F0((__int64)v11, 1, 0, &v68);
      sub_7F6C60((__int64)v14, v68, (__int64)v70);
      v53 = 0;
      v62 = sub_7E2270(*(_QWORD *)(*(_QWORD *)v13 + 8LL));
      v14[5] = (__int64)v62;
LABEL_8:
      v15 = sub_7E7C20((__int64)v7, (__int64)v14, 0, 0);
      v15[11].m128i_i8[1] = 3;
      v16 = v15;
      sub_7EC360((__int64)v15, (__m128i *)((char *)v15 + 177), &v15[11].m128i_i64[1]);
      goto LABEL_9;
    }
    v36 = sub_72D2E0(v7);
    v56 = sub_724EF0(v36);
    **(_QWORD **)v13 = v56;
    v14 = sub_7F54F0((__int64)v11, 1, 0, &v68);
    sub_7F6C60((__int64)v14, v68, (__int64)v70);
    v37 = sub_7E2270(*(_QWORD *)(*(_QWORD *)v13 + 8LL));
    v38 = v56;
    v39 = (__int64)(v14 + 5);
    v53 = 0;
    v62 = v37;
    v14[5] = (__int64)v37;
  }
  v57 = v39;
  v16 = sub_7E2270(v38[1]);
  *(_QWORD *)(*(_QWORD *)v57 + 112LL) = v16;
LABEL_9:
  sub_7E1740(v14[10], (__int64)v69);
  sub_7E2BA0((__int64)v69);
  sub_7FAFA0((__int64)v69);
  if ( a3 )
  {
    v55 = sub_726B30(5);
    v17 = sub_731250(v53);
    v18 = sub_73DBF0(0x24u, (__int64)v63, (__int64)v17);
    *((_QWORD *)v55 + 6) = sub_7F0830(v18);
    v19 = sub_731250((__int64)v62);
    v20 = v6;
    v21 = 0;
    v22 = sub_73DBF0(0x23u, v20, (__int64)v19);
    v64 = (__m128i *)sub_73DCD0(v22);
    if ( a5 )
    {
      v21 = v64;
      v64 = (__m128i *)sub_731250((__int64)v62);
    }
  }
  else
  {
    v40 = sub_73E830((__int64)v62);
    v64 = (__m128i *)sub_73DCD0(v40);
    v55 = 0;
    if ( a5 )
      v21 = (__m128i *)sub_73E830((__int64)v62);
    else
      v21 = 0;
  }
  v23 = sub_73E830((__int64)v16);
  v27 = v23;
  if ( !a4 )
  {
    v41 = sub_73DCD0(v23);
    v27 = sub_731370((__int64)v41, 0, v42, v43, v44, v45);
  }
  v28 = (const __m128i *)sub_698020(v64, 73, (__int64)v27, v24, v25, v26);
  v32 = (__m128i *)v28;
  if ( !a2 && v28[3].m128i_i8[9] == 10 )
    sub_7FA680(v28, 73, v29, a2, v30, v31);
  if ( a5 )
  {
    v33 = sub_7F88E0(a5, v21);
    v32 = (__m128i *)sub_73DF90((__int64)v32, v33);
  }
  sub_7E67B0(v32);
  v34 = sub_732B10((__int64)v32);
  if ( a3 )
  {
    *((_QWORD *)v55 + 9) = v34;
    sub_7E6810((__int64)v55, (__int64)v69, 1);
  }
  else
  {
    sub_7E6810((__int64)v34, (__int64)v69, 1);
  }
  sub_7FB010((__int64)v14, v68, (__int64)v70);
  return v11;
}
