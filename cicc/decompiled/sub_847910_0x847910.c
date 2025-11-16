// Function: sub_847910
// Address: 0x847910
//
void __fastcall sub_847910(
        __int64 a1,
        const __m128i *a2,
        __int64 a3,
        char a4,
        unsigned int a5,
        __int64 *a6,
        _QWORD *a7,
        __m128i *a8)
{
  const __m128i *v8; // r12
  __int64 v10; // r12
  _BOOL4 v11; // ebx
  char v12; // bl
  __int64 v13; // rax
  int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rsi
  __m128i *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r14
  _QWORD *v23; // rbx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rdi
  __int64 v27; // rbx
  _BYTE *v28; // rax
  __int64 v29; // r9
  __int64 v30; // rax
  char v31; // si
  __int64 v32; // rax
  _BYTE *v33; // r8
  __int64 v34; // rax
  unsigned __int8 v35; // bl
  _QWORD *v36; // r14
  unsigned __int8 v37; // di
  __int64 v38; // r12
  __int64 v39; // r14
  _QWORD *v40; // r12
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // rax
  char v44; // dl
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 *v47; // r13
  __int64 v48; // rax
  __int64 *v49; // r13
  _QWORD *v50; // rax
  __int64 v51; // rsi
  void *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  _BYTE *v56; // rax
  unsigned int v58; // [rsp+8h] [rbp-148h]
  int v59; // [rsp+Ch] [rbp-144h]
  __int64 v60; // [rsp+10h] [rbp-140h]
  _BYTE *v61; // [rsp+10h] [rbp-140h]
  __int64 v62; // [rsp+10h] [rbp-140h]
  __int64 v63; // [rsp+10h] [rbp-140h]
  _DWORD *v64; // [rsp+18h] [rbp-138h]
  int v67; // [rsp+30h] [rbp-120h]
  __int64 v68; // [rsp+30h] [rbp-120h]
  __int16 v70; // [rsp+40h] [rbp-110h]
  __int32 v71; // [rsp+44h] [rbp-10Ch]
  __int64 v72; // [rsp+48h] [rbp-108h]
  _BYTE *v73; // [rsp+50h] [rbp-100h]
  int v74; // [rsp+60h] [rbp-F0h]
  char v75; // [rsp+66h] [rbp-EAh]
  _BYTE v76[9]; // [rsp+67h] [rbp-E9h]
  __m128i *v77; // [rsp+70h] [rbp-E0h]
  unsigned int v78; // [rsp+78h] [rbp-D8h]
  int v79; // [rsp+7Ch] [rbp-D4h]
  __int64 v80; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v81; // [rsp+88h] [rbp-C8h]
  char v82; // [rsp+A8h] [rbp-A8h]
  char v83; // [rsp+A9h] [rbp-A7h]
  __m128i v84[10]; // [rsp+B0h] [rbp-A0h] BYREF

  v8 = a2;
  v70 = a5;
  v64 = (_DWORD *)sub_6E1A20(a1);
  v59 = a5 & 1;
  v58 = (a5 >> 9) & 1;
  if ( a8 )
  {
    if ( a6 )
      *a6 = 0;
    sub_82D850((__int64)a8);
    v75 = 0;
LABEL_5:
    if ( dword_4F077C4 != 2 )
      goto LABEL_6;
    goto LABEL_88;
  }
  if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
  {
    v75 = 0;
    if ( !a6 )
      goto LABEL_5;
  }
  else
  {
    v75 = 1;
    if ( !a6 )
      goto LABEL_5;
  }
  *a6 = 0;
  if ( dword_4F077C4 != 2 )
    goto LABEL_6;
LABEL_88:
  if ( (unsigned int)sub_8D23B0(a2) )
    sub_8AE000(a2);
LABEL_6:
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u && !word_4D04898 )
  {
    if ( a8 )
    {
      if ( (unsigned int)sub_8D25A0(a2) )
      {
        v72 = 0;
        v77 = sub_73C570(a2, 1);
        v71 = 1;
        goto LABEL_18;
      }
      v72 = 0;
      v71 = 1;
      goto LABEL_16;
    }
    if ( (unsigned int)sub_6E5430() )
      sub_6851C0(0x1Cu, v64);
    v71 = sub_8D25A0(a2);
    if ( v71 )
    {
      v72 = 0;
      v77 = sub_73C570(a2, 1);
      v71 = 0;
      goto LABEL_103;
    }
    v72 = 0;
    goto LABEL_127;
  }
  v71 = sub_8D3A70(a2);
  if ( !v71 )
  {
    v72 = 0;
    goto LABEL_13;
  }
  v84[0].m128i_i32[0] = 0;
  if ( v75 )
  {
    v72 = sub_6EB250((int)a2, (int)a2, (int)v64, 0, 0);
    if ( a8 )
    {
LABEL_62:
      v71 = v84[0].m128i_i32[0] != 0;
      goto LABEL_13;
    }
    v71 = 0;
  }
  else
  {
    v72 = sub_6EB250((int)a2, (int)a2, (int)v64, 0, (__int64)v84);
    if ( a8 )
      goto LABEL_62;
    v71 = v84[0].m128i_i32[0];
    if ( v84[0].m128i_i32[0] )
    {
      sub_6E50A0();
      v71 = 0;
    }
  }
LABEL_13:
  if ( (unsigned int)sub_8D25A0(a2) )
    goto LABEL_17;
  if ( !a8 )
  {
LABEL_127:
    if ( (unsigned int)sub_6E5430() )
      sub_685360(0x94Au, v64, (__int64)a2);
    goto LABEL_16;
  }
  v71 = 1;
LABEL_16:
  v8 = (const __m128i *)sub_72C930();
LABEL_17:
  v77 = sub_73C570(v8, 1);
  if ( a8 )
  {
LABEL_18:
    v10 = *(_QWORD *)(a1 + 24);
    if ( !v10 )
    {
      *(_QWORD *)&v76[1] = 0;
      goto LABEL_44;
    }
    v73 = 0;
    goto LABEL_20;
  }
LABEL_103:
  v73 = sub_724D50(10);
  v73[169] |= 0x20u;
  v10 = *(_QWORD *)(a1 + 24);
  if ( !v10 )
  {
    v35 = 0;
    v36 = sub_7259C0(8);
    v36[22] = 0;
    v36[20] = v77;
    sub_8D6090(v36);
    v37 = 2;
    *(_QWORD *)&v76[1] = 0;
    *((_QWORD *)v73 + 16) = v36;
    goto LABEL_108;
  }
LABEL_20:
  v79 = 0;
  v74 = 0;
  v67 = 0;
  v76[8] = 0;
  v78 = v70 & 0x4201 | 0x4000000;
  *(_QWORD *)v76 = v72 != 0;
  if ( !*(_QWORD *)(v10 + 16) )
    goto LABEL_32;
LABEL_21:
  v79 = 1;
LABEL_22:
  *(_QWORD *)&v76[1] = 0;
  if ( !dword_4D048B8 )
    goto LABEL_33;
LABEL_23:
  if ( !v72 )
  {
    while ( 1 )
    {
LABEL_33:
      v11 = 0;
      sub_6E6990((__int64)&v80);
      if ( a8 )
        goto LABEL_27;
LABEL_34:
      if ( !v75 )
        v82 |= 0x20u;
      sub_839D30(v10, v77, 0, 1, 0, v78, 0, 0, 0, 0, (char *)&v80, 0);
      v14 = 1;
      if ( (v83 & 0x10) == 0 )
        v14 = v74;
      v74 = v14;
      if ( (v83 & 2) == 0 )
        break;
      v84[0].m128i_i64[0] = (__int64)sub_724DC0();
      sub_72C970(v84[0].m128i_i64[0]);
      v15 = sub_724E50(v84[0].m128i_i64, v77);
      v16 = v15;
      if ( !v75 )
      {
        v63 = v15;
        sub_6E50A0();
        v16 = v63;
      }
      if ( v11 )
        goto LABEL_74;
LABEL_42:
      sub_72A690(v16, (__int64)v73, 0, 0);
      v13 = *(_QWORD *)v10;
      if ( !*(_QWORD *)v10 )
        goto LABEL_43;
LABEL_29:
      if ( *(_BYTE *)(v13 + 8) == 3 )
      {
        v13 = sub_6BBB10((_QWORD *)v10);
        if ( !v13 )
          goto LABEL_43;
      }
      v10 = v13;
      if ( v79 )
        goto LABEL_22;
      if ( *(_QWORD *)(v13 + 16) )
        goto LABEL_21;
LABEL_32:
      ++*(_QWORD *)&v76[1];
      if ( dword_4D048B8 )
        goto LABEL_23;
    }
    if ( v81 )
    {
      v68 = v81;
      v28 = sub_724D50(9);
      v29 = v68;
      v16 = (__int64)v28;
      *((_QWORD *)v28 + 22) = v68;
      *((_QWORD *)v28 + 16) = v77;
      if ( !*(_QWORD *)(v10 + 16) )
        goto LABEL_66;
    }
    else
    {
      v16 = v80;
      if ( !v11 )
        goto LABEL_42;
LABEL_74:
      v61 = (_BYTE *)v16;
      v32 = sub_6EAFA0(2u);
      v33 = v61;
      v62 = v32;
      sub_72F900(v32, v33);
      if ( !v62 )
      {
        v16 = 0;
        goto LABEL_42;
      }
      v56 = sub_724D50(9);
      v29 = v62;
      v16 = (__int64)v56;
      *((_QWORD *)v56 + 22) = v62;
      *((_QWORD *)v56 + 16) = v77;
      if ( !*(_QWORD *)(v10 + 16) )
      {
LABEL_67:
        v30 = qword_4D03C50;
        *(_QWORD *)(v29 + 16) = v72;
        v31 = *(_BYTE *)(v30 + 17);
        if ( v72 && (v31 & 2) != 0 )
        {
          *(_BYTE *)(v72 + 193) |= 0x40u;
          v31 = *(_BYTE *)(v30 + 17);
        }
        v60 = v16;
        sub_734250(v29, v31 & 1);
        v16 = v60;
        v67 = 1;
        goto LABEL_42;
      }
    }
    *(_BYTE *)(v16 + 170) |= 4u;
LABEL_66:
    v67 = 1;
    if ( !v11 )
      goto LABEL_42;
    goto LABEL_67;
  }
  v11 = 0;
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 )
    v11 = *(_BYTE *)(qword_4D03C50 + 16LL) > 3u;
  sub_6E6990((__int64)&v80);
  if ( !a8 )
    goto LABEL_34;
LABEL_27:
  v12 = *(_BYTE *)(v10 + 9);
  *(_BYTE *)(v10 + 9) = v12 & 0xF7;
  sub_839D30(v10, v77, 0, 0, 0, v78, 0, 0, 0, 0, 0, v84);
  *(_BYTE *)(v10 + 9) = (8 * ((v12 & 8) != 0)) | *(_BYTE *)(v10 + 9) & 0xF7;
  if ( v84[0].m128i_i32[2] != 7 )
  {
    sub_832CF0(v84, a8);
    v13 = *(_QWORD *)v10;
    if ( *(_QWORD *)v10 )
      goto LABEL_29;
LABEL_43:
    v10 = 0;
    if ( a8 )
    {
LABEL_44:
      v17 = a3;
      if ( *(_BYTE *)(a3 + 140) == 12 )
        goto LABEL_45;
LABEL_84:
      v17 = a3;
      goto LABEL_46;
    }
    v36 = sub_7259C0(8);
    v35 = v74 & 1;
    v36[20] = v77;
    if ( v79 )
      *((_BYTE *)v36 + 168) |= 0x80u;
    else
      v36[22] = *(_QWORD *)&v76[1];
    sub_8D6090(v36);
    *((_QWORD *)v73 + 16) = v36;
    v37 = v67 == 0 ? 2 : 6;
LABEL_108:
    v38 = sub_6EAFA0(v37);
    sub_72F900(v38, v73);
    *(_BYTE *)(v38 + 50) |= 0x40u;
    if ( v72 )
    {
      *(_QWORD *)(v38 + 16) = v72;
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
        *(_BYTE *)(v72 + 193) |= 0x40u;
    }
    *(_WORD *)(v38 + 50) = *(_WORD *)(v38 + 50) & 0x7F7F | (v35 << 7) | 0x8000;
    v39 = sub_6EC670((__int64)v36, v38, 1, 0);
    if ( v59 )
    {
      sub_82B7B0(v38, v58);
    }
    else if ( (v70 & 0x4000) != 0 )
    {
      if ( v75 && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 && sub_6E53E0(5, 0x946u, v64) )
        sub_684B30(0x946u, v64);
    }
    else if ( (v70 & 0x1000) != 0 )
    {
      v48 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v48 + 4) != 6 || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v48 + 208) + 168LL) + 109LL) & 0x20) == 0)
        && v75
        && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0
        && sub_6E53E0(5, 0x982u, v64) )
      {
        sub_684B30(0x982u, v64);
      }
    }
    v10 = sub_6EE5A0(v39);
    goto LABEL_44;
  }
  v17 = a3;
  v10 = 0;
  v71 = 1;
  if ( *(_BYTE *)(a3 + 140) != 12 )
    goto LABEL_84;
  do
LABEL_45:
    v17 = *(_QWORD *)(v17 + 160);
  while ( *(_BYTE *)(v17 + 140) == 12 );
LABEL_46:
  v18 = 0;
  v19 = *(__m128i **)(*(_QWORD *)(*(_QWORD *)v17 + 96LL) + 8LL);
  v20 = sub_82C1B0((__int64)v19, 0, 0, (__int64)v84);
  if ( !v20 )
  {
LABEL_76:
    v34 = sub_82BD70(v19, v18, v21);
    if ( !*(_QWORD *)(v34 + 1024) || (**(_BYTE **)(v34 + 1008) & 1) == 0 )
    {
      if ( (unsigned int)sub_6E5430() )
        sub_6851C0(0x92Cu, v64);
    }
    if ( !a8 )
      goto LABEL_79;
LABEL_98:
    a8->m128i_i32[2] = 7;
LABEL_59:
    a8[5].m128i_i8[5] |= 0x80u;
    return;
  }
  while ( 1 )
  {
    v22 = *(__int64 **)(v20 + 88);
    v23 = **(_QWORD ***)(v22[19] + 168);
    if ( !v23 || !*v23 || *(_QWORD *)*v23 || !(unsigned int)sub_8D2E30(v23[1]) )
      goto LABEL_48;
    if ( (unsigned int)sub_8D2780(*(_QWORD *)(*v23 + 8LL)) )
      goto LABEL_56;
    v26 = v23[1];
    v18 = *(_QWORD *)(*v23 + 8LL);
    if ( v26 == v18 )
      break;
    if ( (unsigned int)sub_8D97D0(v26, v18, 0, v24, v25) )
    {
LABEL_56:
      v18 = v23[1];
      v27 = *(_QWORD *)(*v23 + 8LL);
      goto LABEL_57;
    }
LABEL_48:
    v19 = v84;
    v20 = sub_82C230(v84);
    if ( !v20 )
      goto LABEL_76;
  }
  v27 = *(_QWORD *)(*v23 + 8LL);
LABEL_57:
  if ( a8 )
  {
    if ( !v71 )
      goto LABEL_59;
    goto LABEL_98;
  }
  v84[0].m128i_i32[0] = 0;
  v40 = sub_73E130((_QWORD *)v10, v18);
  if ( (unsigned int)sub_8D2780(v27) )
  {
    while ( *(_BYTE *)(v27 + 140) == 12 )
      v27 = *(_QWORD *)(v27 + 160);
    v51 = *(unsigned __int8 *)(v27 + 160);
    v52 = sub_73A8E0(*(__int64 *)&v76[1], v51);
  }
  else
  {
    v49 = sub_6EC7D0(v40, 0, v84, 0);
    v50 = sub_73A8E0(*(__int64 *)&v76[1], byte_4F06A51[0]);
    v51 = *v49;
    v49[2] = (__int64)v50;
    v52 = sub_73DBF0(0x32u, v51, (__int64)v49);
  }
  v40[2] = v52;
  sub_6E1D20(v22, v51, v53, v54, v55);
  v41 = sub_6F5430((__int64)v22, (__int64)v40, a3, v58, 1, 0, 0, v84[0].m128i_u32[0], 1u, 0, (__int64)v64);
  if ( sub_730800(v41) )
  {
    if ( a6 )
      *a6 = v41;
LABEL_79:
    if ( a7 )
    {
      sub_6E6260(a7);
      goto LABEL_81;
    }
  }
  else
  {
    *(_BYTE *)(v41 + 51) |= 0x40u;
    *(_BYTE *)(v41 + 50) = (16 * (a4 & 1)) | *(_BYTE *)(v41 + 50) & 0xEF;
    v42 = sub_730250(v41);
    if ( v42 && (*(_BYTE *)(v42 + 171) & 2) != 0 )
    {
      if ( *(_QWORD *)(v42 + 144) )
      {
        v43 = sub_730290(v41);
        v44 = *(_BYTE *)(v43 + 50);
        *(_BYTE *)(v43 + 51) |= 0x40u;
        *(_BYTE *)(v43 + 50) = v44 & 0xEF | (16 * (a4 & 1));
      }
      else
      {
        *(_BYTE *)(v41 + 51) &= ~0x40u;
      }
    }
    v45 = a3;
    if ( *(_BYTE *)(a3 + 140) == 12 )
    {
      do
        v45 = *(_QWORD *)(v45 + 160);
      while ( *(_BYTE *)(v45 + 140) == 12 );
    }
    else
    {
      v45 = a3;
    }
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v45 + 96LL) + 177LL) & 2) == 0 && (unsigned int)sub_6E5430() )
      sub_6851C0(0x943u, v64);
    if ( a6 )
      *a6 = v41;
    if ( a7 )
    {
      v46 = sub_730250(v41);
      if ( v46 && (*(_BYTE *)(v46 + 171) & 2) != 0 )
      {
        sub_6E6A50(v46, (__int64)a7);
      }
      else
      {
        v47 = (__int64 *)sub_6EC670(a3, v41, 0, 0);
        *(__int64 *)((char *)v47 + 28) = *(_QWORD *)v64;
        if ( v59 )
          sub_82B7B0(v41, v58);
        sub_6E70E0(v47, (__int64)a7);
      }
LABEL_81:
      *(_QWORD *)((char *)a7 + 68) = *(_QWORD *)v64;
      *(_QWORD *)((char *)a7 + 76) = *(_QWORD *)sub_6E1A60(a1);
    }
    else if ( v59 )
    {
      sub_82B7B0(v41, v58);
    }
  }
}
