// Function: sub_7E80B0
// Address: 0x7e80b0
//
__int64 __fastcall sub_7E80B0(__int64 *a1, const __m128i *a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 v8; // r12
  _QWORD *v9; // rdi
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __m128i *v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // r9
  __int64 *v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rsi
  void *v19; // rax
  _BYTE *v20; // rax
  __int64 v21; // r12
  __int64 v22; // rax
  const __m128i *v23; // r14
  __int64 v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // rcx
  __m128i *v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  const __m128i *v31; // r14
  _QWORD *v32; // rax
  __int64 v33; // rcx
  __m128i *v34; // rax
  __int64 v35; // rbx
  _QWORD *v36; // rax
  __int64 v37; // rsi
  _BYTE *v38; // rax
  _BYTE *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  _QWORD *v44; // r14
  _QWORD *v45; // rax
  __int64 v46; // rsi
  __m128i *v47; // rax
  void *v48; // r14
  _QWORD *v49; // rax
  __int64 *v50; // rax
  __int64 *v51; // rcx
  __int64 *v52; // r14
  _QWORD *v53; // rax
  __int64 v54; // r14
  __int64 v55; // rbx
  _QWORD *v56; // r14
  __int64 v57; // rsi
  _BYTE *v58; // rax
  _BYTE *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 *v64; // rax
  _BYTE *v65; // rax
  __int64 v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // rax
  __int64 *v69; // rax
  __int64 *v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // rsi
  __int64 *v74; // rax
  _QWORD *v75; // rax
  __int64 v76; // rbx
  __int64 v77; // r14
  _QWORD *v78; // rax
  __int64 v79; // rdx
  __m128i *v80; // rax
  _QWORD *v81; // rax
  __int64 *v82; // [rsp+8h] [rbp-58h]
  __int64 v83; // [rsp+10h] [rbp-50h]
  _QWORD *v84; // [rsp+10h] [rbp-50h]
  __int64 v85; // [rsp+10h] [rbp-50h]
  _BYTE *v86; // [rsp+10h] [rbp-50h]
  _QWORD *v89; // [rsp+20h] [rbp-40h]
  __int64 v90; // [rsp+28h] [rbp-38h]
  __int64 v91; // [rsp+28h] [rbp-38h]
  __int64 v92; // [rsp+28h] [rbp-38h]
  _QWORD *v93; // [rsp+28h] [rbp-38h]
  __int64 *v94; // [rsp+28h] [rbp-38h]
  _QWORD *v95; // [rsp+28h] [rbp-38h]
  __int64 v96; // [rsp+28h] [rbp-38h]
  _QWORD *v97; // [rsp+28h] [rbp-38h]
  __int64 *v98; // [rsp+28h] [rbp-38h]

  v8 = *a1;
  v83 = sub_7E1F20(a2->m128i_i64[0]);
  v9 = (_QWORD *)sub_7E1F00(a2->m128i_i64[0]);
  v10 = sub_72D2E0(v9);
  sub_7E1D00(v9, 0);
  *a3 = sub_7E7CB0(v8);
  v90 = qword_4F189F8;
  v11 = sub_726700(4);
  v12 = *(_QWORD *)(v90 + 120);
  v11[7] = v90;
  *v11 = v12;
  a2[1].m128i_i64[0] = (__int64)v11;
  v13 = sub_73D720(*(const __m128i **)(v90 + 120));
  v14 = (__int64 *)sub_73DBF0(0x5Eu, (__int64)v13, (__int64)a2);
  v15 = a1;
  v16 = v14;
  if ( unk_4F0687C )
  {
    v98 = v14;
    v16[2] = (__int64)sub_73A830(1, byte_4F06A51[0]);
    v71 = (__int64 *)sub_73DBF0(0x36u, *v98, (__int64)v98);
    v15 = a1;
    v16 = v71;
  }
  v91 = (__int64)v16;
  v17 = (__int64 *)sub_7E23D0(v15);
  v18 = *v17;
  v17[2] = v91;
  v19 = sub_73DBF0(0x32u, v18, (__int64)v17);
  v20 = sub_73E110((__int64)v19, v8);
  v21 = sub_7E2BE0(*a3, (__int64)v20);
  if ( unk_4D04768
    && (v22 = *(_QWORD *)(*(_QWORD *)(v83 + 168) + 152LL)) != 0
    && (*(_BYTE *)(v22 + 29) & 0x20) == 0
    && (*(_BYTE *)(v83 + 177) & 1) == 0 )
  {
    v75 = sub_7E8090(a2, 0);
    v76 = qword_4F189F0;
    v77 = (__int64)v75;
    v78 = sub_726700(4);
    v79 = *(_QWORD *)(v76 + 120);
    v78[7] = v76;
    *v78 = v79;
    *(_QWORD *)(v77 + 16) = v78;
    v80 = sub_73D720(*(const __m128i **)(v76 + 120));
    v81 = sub_73DBF0(0x5Eu, (__int64)v80, v77);
    *a5 = (__int64)sub_73E130(v81, v10);
    *a4 = 0;
  }
  else
  {
    v23 = (const __m128i *)sub_7E8090(a2, 0);
    if ( unk_4F0687C )
      v24 = qword_4F189F8;
    else
      v24 = qword_4F189F0;
    v92 = v24;
    v25 = sub_726700(4);
    v26 = *(_QWORD *)(v92 + 120);
    v25[7] = v92;
    *v25 = v26;
    v23[1].m128i_i64[0] = (__int64)v25;
    v27 = sub_73D720(*(const __m128i **)(v92 + 120));
    v93 = sub_73DBF0(0x5Eu, (__int64)v27, (__int64)v23);
    v28 = sub_72BA30(unk_4F06A60);
    v29 = sub_8D6540(v28);
    v94 = (__int64 *)sub_73E130(v93, v29);
    v94[2] = (__int64)sub_7E0E90(1, unk_4F06A60);
    v95 = sub_73DBF0(0x37u, *v94, (__int64)v94);
    v95[2] = sub_7E0E90(0, unk_4F06A60);
    v30 = sub_72BA30(5u);
    v89 = sub_73DBF0(0x3Au, (__int64)v30, (__int64)v95);
    v31 = (const __m128i *)sub_7E8090(v23, 0);
    v96 = qword_4F189F0;
    v32 = sub_726700(4);
    v33 = *(_QWORD *)(v96 + 120);
    v32[7] = v96;
    *v32 = v33;
    v31[1].m128i_i64[0] = (__int64)v32;
    v34 = sub_73D720(*(const __m128i **)(v96 + 120));
    v97 = sub_73DBF0(0x5Eu, (__int64)v34, (__int64)v31);
    v84 = sub_73E830(*a3);
    v35 = sub_7E1DC0();
    v36 = (_QWORD *)sub_7E1C30();
    v37 = sub_72D2E0(v36);
    v38 = sub_73E110((__int64)v84, v37);
    v39 = sub_73DCD0(v38);
    v82 = sub_731370((__int64)v39, v37, v40, v41, v42, v43);
    v44 = sub_7E8090(v31, 0);
    v85 = qword_4F189F0;
    v45 = sub_726700(4);
    v46 = *(_QWORD *)(v85 + 120);
    v45[7] = v85;
    *v45 = v46;
    v44[2] = v45;
    v47 = sub_73D720(*(const __m128i **)(v85 + 120));
    v48 = sub_73DBF0(0x5Eu, (__int64)v47, (__int64)v44);
    v49 = sub_72BA30(unk_4F06A60);
    v50 = (__int64 *)sub_73E110((__int64)v48, (__int64)v49);
    v51 = v82;
    v52 = v50;
    if ( !unk_4F0687C )
    {
      v72 = sub_73A830(1, unk_4F06A60);
      v73 = *v52;
      v52[2] = (__int64)v72;
      v74 = (__int64 *)sub_73DBF0(0x28u, v73, (__int64)v52);
      v51 = v82;
      v52 = v74;
    }
    v51[2] = (__int64)v52;
    v53 = sub_73DBF0(0x32u, *v51, (__int64)v51);
    v86 = sub_73E130(v53, v35);
    v54 = sub_7E7CB0(v35);
    v55 = sub_7E2BE0(v54, (__int64)v86);
    v56 = sub_73E830(v54);
    v57 = sub_72D2E0((_QWORD *)*v97);
    v58 = sub_73E110((__int64)v56, v57);
    v59 = sub_73DCD0(v58);
    v64 = sub_731370((__int64)v59, v57, v60, v61, v62, v63);
    v65 = sub_73DF90(v55, v64);
    v89[2] = v97;
    v97[2] = v65;
    v66 = sub_7E1C50();
    v67 = sub_73DBF0(0x67u, v66, (__int64)v89);
    *a5 = (__int64)sub_73E130(v67, v10);
    v68 = sub_7E7CB0(v10);
    *a4 = v68;
    v69 = (__int64 *)sub_7E2BE0(v68, *a5);
    v21 = (__int64)sub_73DF90(v21, v69);
    *a5 = (__int64)sub_73E830(*a4);
  }
  return v21;
}
