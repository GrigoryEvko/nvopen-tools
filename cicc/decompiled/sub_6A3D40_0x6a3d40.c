// Function: sub_6A3D40
// Address: 0x6a3d40
//
__int64 __fastcall sub_6A3D40(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 *a5, _BOOL4 *a6)
{
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r14
  __int64 i; // rax
  __int64 v18; // rax
  _QWORD *v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdi
  _BOOL4 v27; // edx
  __int64 v28; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // [rsp+8h] [rbp-498h]
  int v38; // [rsp+14h] [rbp-48Ch]
  __int64 v41; // [rsp+40h] [rbp-460h] BYREF
  __int64 v42; // [rsp+48h] [rbp-458h] BYREF
  _QWORD v43[2]; // [rsp+50h] [rbp-450h] BYREF
  _QWORD v44[2]; // [rsp+60h] [rbp-440h] BYREF
  __m128i v45; // [rsp+70h] [rbp-430h]
  __m128i v46; // [rsp+80h] [rbp-420h]
  __m128i v47; // [rsp+90h] [rbp-410h]
  char s[112]; // [rsp+A0h] [rbp-400h] BYREF
  _BYTE v49[160]; // [rsp+110h] [rbp-390h] BYREF
  _BYTE v50[352]; // [rsp+1B0h] [rbp-2F0h] BYREF
  _BYTE v51[400]; // [rsp+310h] [rbp-190h] BYREF

  v41 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  sub_6E1DD0(&v42);
  sub_6E1E00(4, v49, 0, 0);
  sub_72BAF0(v41, a3, unk_4F06A51);
  sub_6F8E70(a1, a4, a4, v50, 0);
  if ( !(unsigned int)sub_8D3070(*(_QWORD *)(a1 + 120)) )
    sub_6ED1A0(v50);
  v9 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v10 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v11 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v12 = *(_QWORD *)a4;
  v44[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v45 = v9;
  v44[1] = v12;
  v46 = v10;
  v47 = v11;
  sub_878540("get", 3u);
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a2) )
    sub_8AE000(a2);
  v13 = (unsigned __int64)v44;
  v16 = sub_7D2AC0(v44, a2, 0);
  if ( !v16 )
    goto LABEL_20;
  v13 = (unsigned int)qword_4F077B4;
  if ( (_DWORD)qword_4F077B4 )
  {
    if ( qword_4F077A0 > 0x1387Fu )
      goto LABEL_7;
  }
  else if ( !HIDWORD(qword_4F077B4) || qword_4F077A8 > 0x1387Fu )
  {
LABEL_7:
    a2 = *(unsigned __int8 *)(v16 + 80);
    i = v16;
    v14 = a2;
    if ( (_BYTE)a2 != 17 )
      goto LABEL_12;
    for ( i = *(_QWORD *)(v16 + 88); i; i = *(_QWORD *)(i + 8) )
    {
      v14 = *(unsigned __int8 *)(i + 80);
LABEL_12:
      v15 = i;
      if ( (_BYTE)v14 == 16 )
      {
        v15 = **(_QWORD **)(i + 88);
        v14 = *(unsigned __int8 *)(v15 + 80);
      }
      if ( (_BYTE)v14 == 24 )
      {
        v15 = *(_QWORD *)(v15 + 88);
        v14 = *(unsigned __int8 *)(v15 + 80);
      }
      if ( (_BYTE)v14 == 20 && (v14 = **(_QWORD **)(*(_QWORD *)(v15 + 88) + 32LL)) != 0 )
      {
        v14 = *(_QWORD *)(v14 + 8);
        if ( *(_BYTE *)(v14 + 80) == 2 )
          goto LABEL_27;
        if ( (_BYTE)a2 != 17 )
          break;
      }
      else if ( (_BYTE)a2 != 17 )
      {
        break;
      }
    }
LABEL_20:
    v18 = sub_6E2EF0(v13, a2, v14, v15);
    v19 = *(_QWORD **)(a1 + 128);
    v20 = v18;
    if ( a3 )
    {
      v21 = 0;
      do
      {
        ++v21;
        v19 = (_QWORD *)*v19;
      }
      while ( a3 != v21 );
    }
    v22 = v41;
    *(_QWORD *)(sub_725090(1) + 48) = v20;
    sub_6E6A50(v22, v20 + 8);
    v23 = sub_6E3060(v50);
    v24 = sub_6E2F40(0);
    *a5 = v24;
    sub_702840("get", *(_QWORD *)(v24 + 24) + 8LL, 0);
    v25 = 0;
    sub_689210(*a5, 0);
    v26 = v23;
    sub_6E1990(v23);
    v27 = 0;
    v28 = *(_QWORD *)(*a5 + 24);
    if ( *(_BYTE *)(v28 + 25) != 1 )
      goto LABEL_24;
    goto LABEL_35;
  }
LABEL_27:
  sub_6E6A50(v41, v51);
  v30 = *(unsigned __int8 *)(v16 + 80);
  if ( (unsigned __int8)v30 <= 0x14u && (v31 = 1182720, _bittest64(&v31, v30)) )
  {
    v32 = sub_6E2EF0(v41, v51, 1182720, v51);
    v33 = sub_725090(1);
    v34 = v41;
    *(_QWORD *)(v33 + 48) = v32;
    v35 = v33;
    sub_6E6A50(v34, v32 + 8);
    v36 = sub_6E2F40(0);
    v25 = (__int64)"get";
    v26 = (__int64)v50;
    *a5 = v36;
    sub_7029D0(v50, "get", v35, 0, v50, *(_QWORD *)(v36 + 24) + 8LL);
  }
  else
  {
    v38 = unk_4D03C70;
    v37 = unk_4D03C78;
    if ( !dword_4D03A1C )
    {
      sub_7ADF70(&unk_4D03A20, 1);
      sub_7CB620("__edg_opnd__(0).get<__edg_opnd__(1)>();", &unk_4D03A20, a4);
      dword_4D03A1C = 1;
    }
    sub_7BC160(&unk_4D03A20);
    v43[0] = v50;
    unk_4D03C70 = 2;
    unk_4D03C78 = v43;
    v43[1] = v51;
    sub_6E1DD0(&v42);
    sub_6E1E00(4, v49, 0, 0);
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x80u;
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    *a5 = sub_6A2B80(0);
    --*(_BYTE *)(qword_4F061C8 + 83LL);
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 )
    {
      sprintf(s, "%lu", a3);
      sub_686470(0xB11u, a4, (__int64)s, *(_QWORD *)(a1 + 120));
      v25 = 65;
      v26 = 75;
      unk_4D03C78 = v37;
      unk_4D03C70 = v38;
      sub_7BE280(75, 65, 0, 0);
      goto LABEL_25;
    }
    v25 = 65;
    v26 = 75;
    unk_4D03C78 = v37;
    unk_4D03C70 = v38;
    sub_7BE280(75, 65, 0, 0);
  }
  v27 = 0;
  v28 = *(_QWORD *)(*a5 + 24);
  if ( *(_BYTE *)(v28 + 25) == 1 )
  {
LABEL_35:
    v26 = v28 + 8;
    v27 = sub_6ED0A0(v28 + 8) == 0;
  }
LABEL_24:
  *a6 = v27;
LABEL_25:
  sub_6E2B30(v26, v25);
  sub_6E1DF0(v42);
  return sub_724E30(&v41);
}
