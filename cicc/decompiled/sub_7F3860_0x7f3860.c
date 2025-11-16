// Function: sub_7F3860
// Address: 0x7f3860
//
void __fastcall sub_7F3860(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 *v8; // r13
  __int64 v9; // rdi
  __int64 k; // rax
  __int64 j; // rbx
  const __m128i *v12; // r14
  __int64 *v13; // rax
  const __m128i *v14; // rax
  const __m128i *v15; // rbx
  __int64 *v16; // r15
  __int64 v17; // rdi
  __int64 m; // r13
  __int64 n; // r14
  _QWORD *v20; // r13
  const __m128i *v21; // r14
  _QWORD *v22; // r15
  _QWORD *v23; // rbx
  _QWORD *v24; // r14
  _QWORD *v25; // rax
  const __m128i *v26; // r13
  __int64 *v27; // rax
  __int64 *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // r15
  __int64 v31; // r14
  __int64 v32; // rbx
  __m128i *v33; // r13
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __m128i *v43; // r15
  _BYTE *v44; // rax
  _BYTE *v45; // rax
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __m128i *v53; // r14
  _BYTE *v54; // r14
  __int64 *v55; // rax
  __int64 v56; // rbx
  __int64 v57; // r13
  __int64 i; // r14
  __m128i *v59; // r15
  char v60; // al
  __int64 v61; // rdx
  __m128i *v62; // r13
  _BYTE *v63; // rax
  __int64 *v64; // rsi
  _BYTE *v65; // r15
  _BYTE *v66; // rax
  __int64 v67; // r13
  __int64 v68; // rax
  _BYTE *v69; // r13
  __int64 *v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // [rsp+8h] [rbp-38h]
  __int64 v73; // [rsp+8h] [rbp-38h]

  v6 = *(_BYTE *)(a1 + 56);
  switch ( v6 )
  {
    case 0u:
      v62 = *(__m128i **)(*(_QWORD *)(a1 + 64) + 16LL);
      if ( dword_4F077C4 == 2 )
      {
        a2 = 0;
        sub_7EE560(v62, 0);
      }
      else
      {
        sub_7D9DD0(*(_QWORD **)(*(_QWORD *)(a1 + 64) + 16LL), (__int64)a2, a3, v6, a5, a6);
      }
      v63 = sub_73E1B0((__int64)v62, (__int64)a2);
      v14 = (const __m128i *)sub_73E110((__int64)v63, *(_QWORD *)a1);
      goto LABEL_10;
    case 0x2Fu:
    case 0x30u:
    case 0x31u:
    case 0x32u:
    case 0x33u:
    case 0x34u:
    case 0x35u:
    case 0x3Au:
    case 0x3Bu:
      sub_7F3600(*(__m128i **)(a1 + 64), 0, 0, 0);
      return;
    case 0x36u:
      v29 = sub_72C610(*(_BYTE *)(*(_QWORD *)a1 + 160LL));
      v30 = *(_QWORD *)(a1 + 64);
      v31 = (__int64)v29;
      v32 = *(_QWORD *)(v30 + 16);
      *(_QWORD *)(v30 + 16) = 0;
      v33 = sub_7E7CA0(*(_QWORD *)a1);
      v34 = sub_731250((__int64)v33);
      v35 = sub_73DC30(0x21u, v31, (__int64)v34);
      v43 = (__m128i *)sub_698020(v35, 73, v30, v36, v37, v38);
      if ( dword_4F077C4 == 2 )
        sub_7EE560(v43, 0);
      else
        sub_7D9DD0(v43, 73, v39, v40, v41, v42);
      v44 = sub_731250((__int64)v33);
      v45 = sub_73DC30(0x22u, v31, (__int64)v44);
      v53 = (__m128i *)sub_698020(v45, 73, v32, v46, v47, v48);
      if ( dword_4F077C4 == 2 )
        sub_7EE560(v53, 0);
      else
        sub_7D9DD0(v53, 73, v49, v50, v51, v52);
      v54 = sub_73DF90((__int64)v43, v53->m128i_i64);
      v55 = sub_73E830((__int64)v33);
      v14 = (const __m128i *)sub_73DF90((__int64)v54, v55);
      goto LABEL_10;
    case 0x3Fu:
      sub_7EE560(*(__m128i **)(a1 + 64), 0);
      v14 = (const __m128i *)sub_73E1B0(*(_QWORD *)(a1 + 64), 0);
      goto LABEL_10;
    case 0x47u:
      v56 = *(_QWORD *)(a1 + 64);
      v57 = *(_QWORD *)(v56 + 56);
      sub_7EAF80(v57, a2);
      sub_7EE560(*(__m128i **)(v56 + 16), 0);
      for ( i = sub_7E7CB0(v57); *(_BYTE *)(v57 + 140) == 12; v57 = *(_QWORD *)(v57 + 160) )
        ;
      v59 = sub_7E7ED0(*(const __m128i **)(v56 + 16));
      v60 = *(_BYTE *)(v57 + 140);
      if ( v60 == 12 )
      {
        v61 = sub_8D4A00(v57);
      }
      else if ( dword_4F077C0 && (v60 == 1 || v60 == 7) )
      {
        v61 = 1;
      }
      else
      {
        v61 = *(_QWORD *)(v57 + 128);
      }
      v73 = v61;
      v64 = (__int64 *)sub_73E230((__int64)v59, 0);
      v65 = sub_73DF90(*(_QWORD *)(v56 + 16), v64);
      v66 = sub_73E230(i, (__int64)v64);
      v67 = sub_7FA4E0(v66, v65, v73, v57);
      v68 = sub_72CBE0();
      v69 = sub_73E110(v67, v68);
      v70 = sub_73E830(i);
      v14 = (const __m128i *)sub_73DF90((__int64)v69, v70);
      goto LABEL_10;
    case 0x4Au:
    case 0x4Bu:
      v8 = *(__int64 **)(a1 + 64);
      if ( v6 == 74 )
      {
        for ( j = v8[7]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v8 = (__int64 *)v8[2];
        v9 = *v8;
      }
      else
      {
        v9 = *v8;
        for ( k = *v8; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        j = *(_QWORD *)(k + 160);
      }
      if ( (unsigned int)sub_7E1F90(v9) || *(char *)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 181LL) >= 0 )
      {
        v12 = (const __m128i *)sub_724D50(1);
        sub_72BB40(*(_QWORD *)a1, v12);
        v13 = (__int64 *)sub_7EBB70((__int64)v12);
        v14 = (const __m128i *)sub_73DF90((__int64)v8, v13);
      }
      else
      {
        v8[2] = (__int64)sub_7E0E90(0, byte_4D03F80[0]);
        v71 = sub_72BA30(5u);
        v14 = (const __m128i *)sub_73DBF0(0x3Au, (__int64)v71, (__int64)v8);
      }
      goto LABEL_10;
    case 0x4Cu:
    case 0x4Du:
      v15 = *(const __m128i **)(a1 + 64);
      v16 = (__int64 *)v15[1].m128i_i64[0];
      if ( v6 == 76 )
      {
        for ( m = v15[3].m128i_i64[1]; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
          ;
        for ( n = v16[7]; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
          ;
        v15 = (const __m128i *)v16[2];
        v16 = (__int64 *)v15[1].m128i_i64[0];
        v17 = v15->m128i_i64[0];
      }
      else
      {
        v17 = v15->m128i_i64[0];
        m = *(_QWORD *)(v15->m128i_i64[0] + 160);
        n = *(_QWORD *)(*v16 + 160);
      }
      if ( (unsigned int)sub_7E1F90(v17)
        || (unsigned int)sub_7E1F90(*v16)
        || *(char *)(*(_QWORD *)(*(_QWORD *)m + 96LL) + 181LL) >= 0
        || *(char *)(*(_QWORD *)(*(_QWORD *)n + 96LL) + 181LL) >= 0 )
      {
        v26 = (const __m128i *)sub_724D50(1);
        sub_72BB40(*(_QWORD *)a1, v26);
        v27 = (__int64 *)sub_7EBB70((__int64)v26);
        v28 = (__int64 *)sub_73DF90((__int64)v16, v27);
        v14 = (const __m128i *)sub_73DF90((__int64)v15, v28);
      }
      else
      {
        v72 = sub_8E4DB0(m, n);
        v20 = sub_72BA30(5u);
        v21 = (const __m128i *)sub_7E8090(v15, 1u);
        v22 = sub_73DBF0(0x3Au, (__int64)v20, (__int64)v15);
        v21[1].m128i_i64[0] = (__int64)sub_7E0E90(-1, byte_4D03F80[0]);
        v23 = sub_73DBF0(0x3Bu, (__int64)v20, (__int64)v21);
        v24 = sub_7E8090(v21, 1u);
        v24[2] = sub_7E0E90(v72, byte_4D03F80[0]);
        v23[2] = sub_73DBF0(0x3Du, (__int64)v20, (__int64)v24);
        v22[2] = sub_73DBF0(0x57u, (__int64)v20, (__int64)v23);
        v25 = sub_73DBF0(0x57u, (__int64)v20, (__int64)v22);
        v14 = (const __m128i *)sub_73E130(v25, *(_QWORD *)a1);
      }
LABEL_10:
      sub_730620(a1, v14);
      return;
    default:
      sub_721090();
  }
}
