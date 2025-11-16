// Function: sub_7F08D0
// Address: 0x7f08d0
//
_QWORD *__fastcall sub_7F08D0(_QWORD *a1, __int64 a2)
{
  __m128i *v2; // r13
  _QWORD *v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // r15
  __int64 v9; // rax
  _BYTE *v10; // r15
  __int64 v11; // rax
  _QWORD *v12; // rbx
  __int64 v13; // rax
  _BYTE *v14; // rbx
  __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  __m128i *v18; // r15
  _BYTE *v19; // rbx
  _QWORD *v20; // r15
  _QWORD *v21; // rax
  const __m128i *v22; // rsi
  __int64 *v24; // r15
  void *v25; // rbx
  _BYTE *v26; // rax
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  _QWORD *v30; // rax
  const __m128i *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  _QWORD *v36; // rax
  _QWORD *v37; // rax
  _BYTE *v38; // rbx
  __int64 v39; // r13
  __int64 *v40; // rax
  _BYTE *v41; // r13
  _QWORD *v42; // rax
  __int64 v43; // rax
  __int64 v44; // [rsp+0h] [rbp-60h]
  __m128i *v45; // [rsp+10h] [rbp-50h]
  _BYTE *v46; // [rsp+10h] [rbp-50h]
  char v47; // [rsp+1Fh] [rbp-41h]
  const __m128i *v48; // [rsp+28h] [rbp-38h] BYREF

  v2 = (__m128i *)a1[9];
  v48 = (const __m128i *)sub_724DC0();
  v47 = *((_BYTE *)a1 + 56);
  if ( v2[1].m128i_i8[8] == 2 && (unsigned int)sub_710600(v2[3].m128i_i64[1]) )
  {
    sub_72BB40(*a1, v48);
    v36 = sub_73A720(v48, (__int64)v48);
    *(_QWORD *)((char *)v36 + 28) = *(_QWORD *)((char *)a1 + 28);
    sub_730620((__int64)a1, (const __m128i *)v36);
    return sub_724E30((__int64)&v48);
  }
  v3 = (_QWORD *)*a1;
  v4 = v2->m128i_i64[0];
  if ( v47 == 19 )
  {
    a2 = 0;
    v5 = *a1;
    v3 = (_QWORD *)sub_72D2E0(v3);
  }
  else
  {
    v5 = sub_8D46C0(*a1);
    v4 = sub_8D46C0(v4);
  }
  while ( *(_BYTE *)(v5 + 140) == 12 )
    v5 = *(_QWORD *)(v5 + 160);
  for ( ; *(_BYTE *)(v4 + 140) == 12; v4 = *(_QWORD *)(v4 + 160) )
    ;
  if ( v47 != 19 )
  {
    v45 = (__m128i *)sub_7E8090(v2, 0);
    if ( !(unsigned int)sub_8D2600(v5) )
      goto LABEL_10;
LABEL_16:
    v24 = sub_7E8090(v45, 0);
    if ( (unsigned int)sub_8D2E30(*v24) )
      v24 = (__int64 *)sub_73DCD0(v24);
    v25 = sub_7E4640(v24, -2);
    v26 = sub_7E23D0(v45);
    *((_QWORD *)v26 + 2) = v25;
    v27 = (__int64)v26;
    v28 = sub_7E1C30();
    v18 = (__m128i *)sub_73DBF0(0x32u, v28, v27);
    if ( v47 != 19 )
      goto LABEL_13;
    v29 = 0;
LABEL_20:
    v30 = sub_73DBF0(0x67u, (__int64)v3, v29);
    v31 = (const __m128i *)sub_73DCD0(v30);
    v22 = v31;
    if ( (*((_BYTE *)a1 + 25) & 1) == 0 )
      v22 = (const __m128i *)sub_731370((__int64)v31, (__int64)v31, v32, v33, v34, v35);
    goto LABEL_14;
  }
  v45 = (__m128i *)sub_73E1B0((__int64)v2, a2);
  v2 = v45;
  if ( (unsigned int)sub_8D2600(v5) )
    goto LABEL_16;
LABEL_10:
  v6 = sub_7E1C10();
  v46 = sub_73E110((__int64)v45, v6);
  v7 = sub_7DDA20(v5);
  sub_72D510(v7, (__int64)v48, 1);
  v8 = sub_73A720(v48, (__int64)v48);
  v9 = sub_7DB8E0(v48, v48);
  v10 = sub_73E110((__int64)v8, v9);
  v11 = sub_7DDA20(v4);
  sub_72D510(v11, (__int64)v48, 1);
  v12 = sub_73A720(v48, (__int64)v48);
  v13 = sub_7DB8E0(v48, v48);
  v14 = sub_73E110((__int64)v12, v13);
  v15 = unk_4F06A60;
  v16 = sub_73A830(-1, unk_4F06A60);
  *((_QWORD *)v46 + 2) = v14;
  *((_QWORD *)v14 + 2) = v10;
  *((_QWORD *)v10 + 2) = v16;
  sub_7DB8E0(-1, v15);
  sub_7DB8E0(-1, v15);
  v17 = sub_72BA30(unk_4F06A60);
  if ( qword_4F189C8 )
  {
    v18 = (__m128i *)sub_7F88E0(qword_4F189C8, v46);
  }
  else
  {
    v44 = (__int64)v17;
    sub_7E1C10();
    sub_7E1C10();
    v18 = (__m128i *)sub_7F8AB0("__dynamic_cast", v44, 0, 0, 0, (__int64)v46);
  }
  if ( v47 == 19 )
  {
    v37 = sub_7E8090(v18, 0);
    v38 = sub_73E130(v37, (__int64)v3);
    if ( qword_4F189D0 )
    {
      v39 = sub_7F88E0(qword_4F189D0, 0);
    }
    else
    {
      v43 = sub_72CBE0();
      v39 = sub_7F8B20("__cxa_bad_cast", &qword_4F189D0, v43, 0, 0, 0);
    }
    sub_72BB40((__int64)v3, v48);
    v40 = sub_73A720(v48, (__int64)v48);
    v41 = sub_73DF90(v39, v40);
    v42 = sub_7F0830(v18);
    v42[2] = v38;
    v29 = (__int64)v42;
    *((_QWORD *)v38 + 2) = v41;
    goto LABEL_20;
  }
LABEL_13:
  v19 = sub_73E130(v18, (__int64)v3);
  sub_72BB40((__int64)v3, v48);
  v20 = sub_73A720(v48, (__int64)v48);
  v21 = sub_7F0830(v2);
  v21[2] = v19;
  *((_QWORD *)v19 + 2) = v20;
  v22 = (const __m128i *)sub_73DBF0(0x67u, (__int64)v3, (__int64)v21);
LABEL_14:
  sub_730620((__int64)a1, v22);
  return sub_724E30((__int64)&v48);
}
