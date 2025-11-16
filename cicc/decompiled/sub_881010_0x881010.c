// Function: sub_881010
// Address: 0x881010
//
__int64 __fastcall sub_881010(_BYTE *src, size_t a2, __int64 *a3, _QWORD *a4, int a5, _QWORD *a6)
{
  size_t v6; // r10
  char v10; // al
  int v11; // r13d
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // r14d
  __m128i v18; // xmm1
  __m128i v19; // xmm2
  __m128i v20; // xmm3
  __int64 v21; // r8
  __int64 v22; // r15
  char v23; // al
  __int64 v24; // rcx
  int v25; // r12d
  char v26; // si
  char v27; // dl
  __int64 v28; // r13
  size_t v29; // r15
  size_t v30; // rcx
  int v31; // r10d
  __int64 v32; // rdx
  _QWORD *v33; // rax
  size_t v34; // r10
  int v35; // eax
  bool v36; // zf
  int v37; // eax
  _QWORD *v38; // rax
  _QWORD *v39; // r12
  __int64 v41; // rax
  __int64 i; // r11
  _QWORD *v43; // r10
  int v44; // eax
  int v45; // eax
  _QWORD *v46; // rdi
  char v47; // al
  int v48; // eax
  _QWORD *v49; // rax
  __int64 j; // rax
  _QWORD *v51; // r12
  _QWORD *v52; // rax
  int v53; // eax
  const __m128i *v54; // rax
  __m128i *v55; // rax
  int v56; // eax
  char v57; // [rsp+16h] [rbp-CAh]
  size_t v58; // [rsp+18h] [rbp-C8h]
  _QWORD *v59; // [rsp+20h] [rbp-C0h]
  size_t v60; // [rsp+20h] [rbp-C0h]
  size_t v61; // [rsp+20h] [rbp-C0h]
  __int64 v62; // [rsp+28h] [rbp-B8h]
  _QWORD *v63; // [rsp+28h] [rbp-B8h]
  size_t v64; // [rsp+28h] [rbp-B8h]
  __int64 v65; // [rsp+28h] [rbp-B8h]
  int v66; // [rsp+30h] [rbp-B0h]
  int v67; // [rsp+34h] [rbp-ACh]
  __int64 v68; // [rsp+38h] [rbp-A8h]
  _QWORD *v69; // [rsp+40h] [rbp-A0h]
  _QWORD *v70; // [rsp+48h] [rbp-98h]
  size_t v71; // [rsp+50h] [rbp-90h]
  int v72; // [rsp+58h] [rbp-88h]
  int v73; // [rsp+5Ch] [rbp-84h]
  __int64 v74; // [rsp+60h] [rbp-80h]
  size_t n; // [rsp+68h] [rbp-78h]
  size_t na; // [rsp+68h] [rbp-78h]
  size_t nb; // [rsp+68h] [rbp-78h]
  __m128i v78[7]; // [rsp+70h] [rbp-70h] BYREF

  v6 = a2;
  v66 = a5;
  if ( !a5 )
  {
    sub_724C70((__int64)xmmword_4F06220, 0);
    v6 = a2;
  }
  v10 = *((_BYTE *)a4 + 140);
  if ( v10 != 2 )
  {
    n = v6;
    if ( v10 != 3 )
    {
      v11 = 1;
      v12 = (_QWORD *)sub_8D4050(a4);
      v13 = sub_72D2E0(v12);
      v6 = n;
      a4 = (_QWORD *)v13;
      goto LABEL_6;
    }
    v33 = sub_72C610(6u);
    v34 = n;
    a4 = v33;
LABEL_28:
    v17 = 1;
    sub_87A880(src, v34, (__m128i *)&qword_4D04A00, a3);
    v11 = dword_4F04D80;
    if ( !dword_4F04D80 )
      goto LABEL_7;
    if ( v66 )
    {
LABEL_31:
      v11 = 0;
      goto LABEL_7;
    }
LABEL_30:
    sub_877830();
    goto LABEL_31;
  }
  v11 = 0;
  if ( *((_BYTE *)a4 + 160) && (a4[20] & 0x3C000) == 0 )
  {
    nb = v6;
    v52 = sub_72BA30(0xAu);
    v34 = nb;
    a4 = v52;
    goto LABEL_28;
  }
LABEL_6:
  sub_87A880(src, v6, (__m128i *)&qword_4D04A00, a3);
  v17 = dword_4F04D80;
  if ( dword_4F04D80 )
  {
    v17 = unk_4F07700;
    if ( !unk_4F07700 )
    {
      if ( dword_4F077C4 != 2 || unk_4F07778 <= 202001 || (v11 & 1) == 0 )
        goto LABEL_7;
      if ( v66 )
      {
        v11 = 1;
        goto LABEL_7;
      }
      goto LABEL_95;
    }
    v17 = 0;
    if ( v66 )
      goto LABEL_7;
    if ( v11 )
    {
LABEL_95:
      v17 = 0;
      sub_72A510(xmmword_4F06300, xmmword_4F06220);
      goto LABEL_7;
    }
    goto LABEL_30;
  }
LABEL_7:
  v18 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
  v19 = _mm_loadu_si128(&xmmword_4D04A20);
  v20 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  v78[0] = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  v78[1] = v18;
  v78[2] = v19;
  v78[3] = v20;
  v22 = sub_7D5DD0(v78, 0, v14, v15, v16);
  if ( !v22 )
  {
    v74 = 0;
    v70 = 0;
    if ( a6 )
      goto LABEL_47;
    return 0;
  }
  if ( !qword_4F5FFA8 )
  {
    qword_4F5FFA8 = (__int64)sub_72BA30(byte_4F06A51[0]);
    v54 = (const __m128i *)sub_72BA30(0);
    v55 = sub_73C570(v54, 1);
    qword_4F5FFA0 = sub_72D2E0(v55);
  }
  v23 = *(_BYTE *)(v22 + 80);
  v24 = v22;
  if ( v23 == 17 )
  {
    v24 = *(_QWORD *)(v22 + 88);
    v23 = *(_BYTE *)(v24 + 80);
  }
  v69 = a4;
  v25 = v11;
  v26 = v11 & 1;
  v27 = v11 ^ 1;
  v28 = v22;
  v74 = 0;
  v29 = v24;
  v70 = 0;
  v67 = 0;
  v68 = 0;
  v73 = 0;
  na = 0;
  v72 = 0;
  v71 = 0;
  v57 = v27 & 1;
  while ( 1 )
  {
    v30 = v29;
    if ( v23 == 16 )
    {
      v30 = **(_QWORD **)(v29 + 88);
      v23 = *(_BYTE *)(v30 + 80);
    }
    if ( v23 == 24 )
    {
      v30 = *(_QWORD *)(v30 + 88);
      v23 = *(_BYTE *)(v30 + 80);
    }
    if ( v23 == 20 )
      break;
    if ( v23 != 11 )
      goto LABEL_14;
    v41 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v30 + 88) + 152LL) + 168LL);
    if ( !v41 )
      goto LABEL_14;
    for ( i = *(_QWORD *)(v41 + 8); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v43 = *(_QWORD **)v41;
    if ( *(_QWORD *)v41 )
    {
      for ( j = v43[1]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      if ( *v43 )
        goto LABEL_14;
      v43 = (_QWORD *)j;
    }
    if ( (qword_4F5FFA0 == i
       || (v58 = v30,
           v59 = v43,
           v62 = i,
           v44 = sub_8D97D0(i, qword_4F5FFA0, 0, v30, v21),
           i = v62,
           v43 = v59,
           v30 = v58,
           v44))
      && !v43
      && (v17 & 1) != 0 )
    {
      v48 = 1;
      if ( !v71 )
        v48 = v72;
      v71 = v30;
      v72 = v48;
      if ( a6 )
      {
        v49 = sub_878440();
        v49[1] = v71;
        *v49 = v74;
        v74 = (__int64)v49;
      }
    }
    else if ( (_QWORD *)i == v69
           || (v60 = v30, v63 = v43, v45 = sub_8D97D0(v69, i, 0, v30, v21), v43 = v63, v30 = v60, v45) )
    {
      if ( v43 || !v26 )
      {
        if ( v25 )
        {
          if ( (_QWORD *)qword_4F5FFA8 != v43 )
          {
            v64 = v30;
            if ( !(unsigned int)sub_8DED30(v43, qword_4F5FFA8, 1) )
              goto LABEL_14;
            v30 = v64;
          }
        }
        else if ( v43 )
        {
          goto LABEL_14;
        }
        if ( v68 )
        {
          if ( !a6 )
            return v28;
          v67 = 1;
LABEL_69:
          v68 = v30;
          v46 = v70;
          v70 = sub_878440();
          v70[1] = v68;
          *v70 = v46;
          goto LABEL_14;
        }
        if ( a6 )
          goto LABEL_69;
        v68 = v30;
      }
    }
LABEL_14:
    if ( *(_BYTE *)(v28 + 80) != 17 )
      goto LABEL_40;
LABEL_15:
    v29 = *(_QWORD *)(v29 + 8);
    if ( !v29 )
      goto LABEL_40;
    v23 = *(_BYTE *)(v29 + 80);
  }
  v31 = unk_4F07700;
  if ( v17 | unk_4F07700 )
  {
    v32 = **(_QWORD **)(*(_QWORD *)(v30 + 88) + 328LL);
    if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 )
      goto LABEL_71;
    goto LABEL_33;
  }
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 202001 || v57 )
    goto LABEL_14;
  v32 = **(_QWORD **)(*(_QWORD *)(v30 + 88) + 328LL);
LABEL_71:
  v47 = *(_BYTE *)(*(_QWORD *)(v32 + 8) + 80LL);
  if ( !*(_QWORD *)v32 )
  {
    if ( (*(_BYTE *)(v32 + 56) & 0x10) != 0 || v47 != 2 )
      goto LABEL_34;
    v61 = v30;
    v65 = v32;
    v53 = sub_8D3A70(*(_QWORD *)(*(_QWORD *)(v32 + 64) + 128LL));
    v30 = v61;
    if ( v53 || (v56 = sub_8D3F60(*(_QWORD *)(*(_QWORD *)(v65 + 64) + 128LL)), v32 = v65, v30 = v61, v56) )
    {
      v35 = 1;
      goto LABEL_35;
    }
LABEL_33:
    if ( *(_BYTE *)(*(_QWORD *)(v32 + 8) + 80LL) == 3 && *(_QWORD *)v32 )
    {
      v31 = unk_4F07700;
      goto LABEL_51;
    }
    goto LABEL_34;
  }
  if ( v47 == 3 )
  {
LABEL_51:
    v35 = v31 != 0;
    goto LABEL_35;
  }
LABEL_34:
  v35 = 0;
LABEL_35:
  if ( v35 != v25 )
    goto LABEL_14;
  v36 = na == 0;
  v37 = 1;
  na = v30;
  if ( v36 )
    v37 = v73;
  v73 = v37;
  if ( !a6 )
    goto LABEL_14;
  v38 = sub_878440();
  v38[1] = na;
  *v38 = v74;
  v74 = (__int64)v38;
  if ( *(_BYTE *)(v28 + 80) == 17 )
    goto LABEL_15;
LABEL_40:
  v22 = v28;
  if ( v67 )
  {
    if ( !a6 )
      return v22;
    if ( v70 )
    {
      v51 = v70;
      do
      {
        sub_67E1D0(a6, 421, v51[1]);
        v51 = (_QWORD *)*v51;
      }
      while ( v51 );
    }
    goto LABEL_47;
  }
  if ( !v68 )
  {
    if ( na )
    {
      if ( v71 || v73 )
      {
        if ( v66 )
          goto LABEL_43;
        v66 = 1;
      }
      else
      {
        if ( v66 )
          goto LABEL_113;
        v22 = na;
      }
    }
    else
    {
      if ( !v71 )
      {
        v22 = 0;
        if ( !a6 )
          return 0;
        goto LABEL_44;
      }
      if ( v72 )
      {
        if ( v66 )
          goto LABEL_43;
        v66 = v72;
      }
      else
      {
        v22 = v71;
        if ( v66 )
        {
          na = v71;
LABEL_113:
          if ( unk_4F062CD )
          {
            sub_72A510(xmmword_4F06220, xmmword_4F06300);
            v22 = na;
            if ( !a6 )
              return v22;
          }
          else
          {
            v22 = na;
            if ( !a6 )
              return v22;
          }
          goto LABEL_44;
        }
      }
    }
    if ( dword_4F04D80 )
      goto LABEL_43;
    if ( v25 )
      sub_72A510(xmmword_4F06300, xmmword_4F06220);
    else
      sub_877830();
    if ( v66 )
      goto LABEL_43;
    sub_72A510(xmmword_4F06220, xmmword_4F06300);
    if ( !a6 )
      return v22;
    goto LABEL_44;
  }
  v22 = v68;
LABEL_43:
  if ( !a6 )
    return v22;
LABEL_44:
  if ( v74 )
  {
    v39 = (_QWORD *)v74;
    do
    {
      sub_67E1D0(a6, 421, v39[1]);
      v39 = (_QWORD *)*v39;
    }
    while ( v39 );
  }
LABEL_47:
  sub_878490((__int64)v70);
  sub_878490(v74);
  return v22;
}
