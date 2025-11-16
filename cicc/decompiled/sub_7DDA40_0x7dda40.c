// Function: sub_7DDA40
// Address: 0x7dda40
//
_QWORD *__fastcall sub_7DDA40(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // r12
  int v9; // r14d
  int v10; // ebx
  _BYTE *v11; // r14
  __int64 v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdi
  _BYTE *i; // r12
  const __m128i *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  const __m128i *v24; // rsi
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  _QWORD *v29; // r12
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 *v32; // rbx
  __int64 v33; // r12
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rax
  _BYTE *v37; // r12
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rax
  _BYTE *v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // rbx
  __int64 j; // rsi
  _BYTE *v46; // rax
  const __m128i *v47; // [rsp+8h] [rbp-38h] BYREF

  v47 = (const __m128i *)sub_724DC0();
  v7 = *(_QWORD *)(a1 + 56);
  if ( (*(_BYTE *)(a1 + 64) & 1) != 0 )
  {
    v8 = *(_QWORD *)(v7 + 16);
    v9 = HIDWORD(qword_4F077B4);
    if ( HIDWORD(qword_4F077B4) )
    {
      v9 = dword_4F06970;
      dword_4F06970 = 1;
    }
    v10 = sub_7311F0(v8, a2, v3, v4, v5, v6);
    if ( HIDWORD(qword_4F077B4) )
      dword_4F06970 = v9;
    sub_7EE560(v8, 0);
    if ( *(_BYTE *)(v8 + 24) == 1 && *(_BYTE *)(v8 + 56) == 92 )
    {
      v11 = *(_BYTE **)(v8 + 72);
      if ( (*(_BYTE *)(v8 + 59) & 0x20) != 0 )
        v11 = (_BYTE *)*((_QWORD *)v11 + 2);
      *((_QWORD *)v11 + 2) = 0;
    }
    else
    {
      v11 = sub_73E1B0(v8, 0);
    }
    if ( v10 )
    {
      v12 = sub_7E4620(v11, 0);
      *(_QWORD *)(v12 + 16) = sub_73A830(-1, 5u);
      v13 = sub_7E1330();
      v14 = sub_73DBF0(0x5Cu, v13, v12);
      v15 = (_QWORD *)sub_7DBE60();
      v16 = sub_72D2E0(v15);
      v17 = sub_73E130(v14, v16);
    }
    else
    {
      v26 = sub_7E8090(v11, 0);
      v27 = sub_7E4620(v26, 0);
      *(_QWORD *)(v27 + 16) = sub_73A830(-1, 5u);
      v28 = sub_7E1330();
      v29 = sub_73DBF0(0x5Cu, v28, v27);
      v30 = (_QWORD *)sub_7DBE60();
      v31 = sub_72D2E0(v30);
      v32 = (__int64 *)sub_73E130(v29, v31);
      sub_72CBE0();
      sub_7F8110("__cxa_bad_typeid", 0, 0, 0, 0);
      v33 = sub_7F88E0(qword_4F18938, 0);
      v34 = (_QWORD *)sub_7DBE60();
      v35 = sub_72D2E0(v34);
      sub_72BB40(v35, v47);
      v36 = sub_73A720(v47, (__int64)v47);
      v37 = sub_73DF90(v33, v36);
      v38 = sub_7F0830(v11);
      *(_QWORD *)(v38 + 16) = v32;
      v39 = *v32;
      v32[2] = (__int64)v37;
      v17 = sub_73DBF0(0x67u, v39, v38);
    }
    i = sub_73DCD0(v17);
  }
  else
  {
    v40 = sub_7DDA20(*(_QWORD *)(v7 + 56));
    v41 = sub_731250(v40);
    v44 = *(_QWORD *)v41;
    for ( i = v41; *(_BYTE *)(v44 + 140) == 12; v44 = *(_QWORD *)(v44 + 160) )
      ;
    for ( j = qword_4F18960[0]; qword_4F18960[0] != v44; j = qword_4F18960[0] )
    {
      if ( (unsigned int)sub_8D97D0(v44, j, 0, v42, v43) )
        break;
      v46 = sub_73DE50((__int64)i, *(_QWORD *)(v44 + 160));
      v44 = *(_QWORD *)v46;
      for ( i = v46; *(_BYTE *)(v44 + 140) == 12; v44 = *(_QWORD *)(v44 + 160) )
        ;
    }
  }
  v19 = (const __m128i *)sub_73DC90(i, *(_QWORD *)a1);
  v24 = v19;
  if ( (*(_BYTE *)(a1 + 25) & 1) == 0 )
    v24 = (const __m128i *)sub_731370((__int64)v19, (__int64)v19, v20, v21, v22, v23);
  sub_730620(a1, v24);
  return sub_724E30((__int64)&v47);
}
