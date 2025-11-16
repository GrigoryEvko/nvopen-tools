// Function: sub_37A67D0
// Address: 0x37a67d0
//
unsigned __int8 *__fastcall sub_37A67D0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rsi
  unsigned __int16 *v4; // rax
  unsigned __int16 v5; // bx
  const __m128i *v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rsi
  int v9; // r9d
  unsigned int v10; // edx
  __int64 v11; // rax
  unsigned __int16 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdx
  char v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  int v20; // eax
  unsigned __int8 *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r13
  unsigned __int16 v25; // r15
  __int64 v26; // r12
  unsigned __int16 v27; // ax
  _BYTE *v28; // rbx
  __int16 v29; // r14
  __int64 v30; // r12
  __int16 v31; // r11
  __int64 v32; // rax
  __int64 v33; // rdx
  char v34; // di
  int v35; // edx
  unsigned __int16 v36; // bx
  unsigned int v37; // r15d
  unsigned int v38; // r14d
  unsigned int v39; // eax
  __int64 v40; // rcx
  __int64 v41; // rax
  unsigned __int16 v42; // r12
  __int64 v43; // rax
  __int64 v44; // r13
  __int64 v45; // rdx
  char v46; // bl
  _BYTE *v47; // rdx
  _BYTE *v48; // rax
  __int64 v49; // rdx
  _BYTE *v51; // rdx
  _BYTE *v52; // rax
  _QWORD *v53; // r12
  __int128 v54; // rax
  _QWORD *v55; // rdi
  __int128 v56; // rax
  __int64 v57; // r9
  unsigned int v58; // edx
  __int128 v59; // rax
  __int64 v60; // r9
  unsigned int v61; // edx
  __int64 v62; // [rsp+0h] [rbp-130h]
  int v63; // [rsp+Ch] [rbp-124h]
  __int64 v64; // [rsp+18h] [rbp-118h]
  __int64 v65; // [rsp+20h] [rbp-110h]
  unsigned int v68; // [rsp+38h] [rbp-F8h]
  __int16 v69; // [rsp+3Ch] [rbp-F4h]
  unsigned __int16 v70; // [rsp+3Eh] [rbp-F2h]
  __int128 v71; // [rsp+40h] [rbp-F0h]
  __int128 v72; // [rsp+50h] [rbp-E0h]
  _BYTE *v73; // [rsp+60h] [rbp-D0h]
  __int64 v74; // [rsp+70h] [rbp-C0h]
  __int128 v75; // [rsp+70h] [rbp-C0h]
  __int64 v76; // [rsp+80h] [rbp-B0h] BYREF
  int v77; // [rsp+88h] [rbp-A8h]
  unsigned int v78; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v79; // [rsp+98h] [rbp-98h]
  unsigned __int16 v80; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v81; // [rsp+A8h] [rbp-88h]
  __int64 v82; // [rsp+B0h] [rbp-80h]
  __int64 v83; // [rsp+B8h] [rbp-78h]
  __int64 v84; // [rsp+C0h] [rbp-70h]
  __int64 v85; // [rsp+C8h] [rbp-68h]
  __int64 v86; // [rsp+D0h] [rbp-60h]
  __int64 v87; // [rsp+D8h] [rbp-58h]
  __int64 v88; // [rsp+E0h] [rbp-50h]
  _BYTE *v89; // [rsp+E8h] [rbp-48h]
  __int64 v90; // [rsp+F0h] [rbp-40h] BYREF
  __int64 v91; // [rsp+F8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 80);
  v76 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v76, v3, 1);
  v77 = *(_DWORD *)(a2 + 72);
  v4 = *(unsigned __int16 **)(a2 + 48);
  v5 = *v4;
  v79 = *((_QWORD *)v4 + 1);
  v6 = *(const __m128i **)(a2 + 40);
  LOWORD(v78) = v5;
  v7 = _mm_loadu_si128(v6);
  v8 = sub_379AB60((__int64)a1, v7.m128i_u64[0], v7.m128i_i64[1]);
  v65 = v8;
  *(_QWORD *)&v71 = v8;
  *((_QWORD *)&v71 + 1) = v10 | v7.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v64 = 16LL * v10;
  v11 = *(_QWORD *)(v8 + 48) + v64;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v80 = v12;
  v81 = v13;
  if ( v5 )
  {
    if ( v5 == 1 || (unsigned __int16)(v5 - 504) <= 7u )
      goto LABEL_73;
    v14 = *(_QWORD *)&byte_444C4A0[16 * v5 - 16];
    v16 = byte_444C4A0[16 * v5 - 8];
  }
  else
  {
    v84 = sub_3007260((__int64)&v78);
    v14 = v84;
    v85 = v15;
    v16 = v15;
  }
  if ( v12 )
  {
    if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      goto LABEL_73;
    v51 = &byte_444C4A0[16 * v12 - 16];
    if ( *(_QWORD *)v51 == v14 && v16 == v51[8] )
      goto LABEL_8;
    v62 = 0;
    v69 = word_4456580[v12 - 1];
  }
  else
  {
    v82 = sub_3007260((__int64)&v80);
    v83 = v17;
    if ( v82 == v14 && (_BYTE)v17 == v16 )
      goto LABEL_8;
    v69 = sub_3009970((__int64)&v80, v8, v17, v18, v19);
    v62 = v22;
  }
  v23 = v2;
  v24 = 17;
  v63 = v5;
  v25 = v5;
  v26 = 16LL * (v5 - 1);
  v27 = v5 - 504;
  v28 = byte_444C4A0;
  v73 = &byte_444C4A0[v26];
  v29 = 2;
  v70 = v27;
  v30 = *a1;
  while ( 1 )
  {
    v31 = v24;
    LOWORD(v23) = v24;
    if ( *(_QWORD *)(v30 + 8 * v24 + 112) )
    {
      if ( v25 )
      {
        if ( v70 <= 7u )
          goto LABEL_73;
        v32 = *(_QWORD *)v73;
        v34 = v73[8];
      }
      else
      {
        v74 = v23;
        v32 = sub_3007260((__int64)&v78);
        v23 = v74;
        v31 = v24;
        v86 = v32;
        v87 = v33;
        v34 = v33;
      }
      v35 = v24 - 1;
      if ( v32 == *((_QWORD *)v28 + 32) && v28[264] == v34 && v69 == v29 && (v69 || !v62) )
        break;
    }
    v28 += 16;
    if ( v24 == 228 )
    {
      v36 = v25;
      goto LABEL_32;
    }
    v29 = word_4456580[v24++];
  }
  v36 = v25;
  v37 = v23;
  if ( (unsigned __int16)(v31 - 176) <= 0x34u )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
    v35 = v24 - 1;
  }
  v38 = word_4456340[v35];
  if ( !v80 )
  {
    if ( !sub_3007100((__int64)&v80) )
      goto LABEL_58;
    goto LABEL_66;
  }
  if ( (unsigned __int16)(v80 - 176) <= 0x34u )
  {
LABEL_66:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v80 )
    {
      if ( (unsigned __int16)(v80 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_26;
    }
LABEL_58:
    v39 = sub_3007130((__int64)&v80, (__int64)byte_444C4A0);
    goto LABEL_59;
  }
LABEL_26:
  v39 = word_4456340[v80 - 1];
LABEL_59:
  v53 = (_QWORD *)a1[1];
  if ( v39 >= v38 )
  {
    *(_QWORD *)&v59 = sub_3400EE0((__int64)v53, 0, (__int64)&v76, 0, v7);
    v65 = (__int64)sub_3406EB0(v53, 0xA1u, (__int64)&v76, v37, 0, v60, v71, v59);
    v68 = v61;
  }
  else
  {
    *(_QWORD *)&v54 = sub_3400EE0((__int64)v53, 0, (__int64)&v76, 0, v7);
    v55 = (_QWORD *)a1[1];
    v75 = v54;
    v90 = 0;
    LODWORD(v91) = 0;
    *(_QWORD *)&v56 = sub_33F17F0(v55, 51, (__int64)&v90, v37, 0);
    if ( v90 )
    {
      v72 = v56;
      sub_B91220((__int64)&v90, v90);
      v56 = v72;
    }
    v65 = sub_340F900(v53, 0xA0u, (__int64)&v76, v37, 0, v57, v56, v71, v75);
    v68 = v58;
  }
  v64 = 16LL * v68;
LABEL_32:
  v40 = v65;
  v41 = *(_QWORD *)(v65 + 48) + v64;
  v42 = *(_WORD *)v41;
  v43 = *(_QWORD *)(v41 + 8);
  v80 = v42;
  v81 = v43;
  if ( v36 )
  {
    if ( (unsigned __int16)(v36 - 504) <= 7u )
      goto LABEL_73;
    v52 = &byte_444C4A0[16 * v63 - 16];
    v44 = *(_QWORD *)v52;
    v46 = v52[8];
  }
  else
  {
    v90 = sub_3007260((__int64)&v78);
    v44 = v90;
    v91 = v45;
    v46 = v45;
  }
  if ( v42 )
  {
    if ( v42 == 1 || (unsigned __int16)(v42 - 504) <= 7u )
      goto LABEL_73;
    v48 = &byte_444C4A0[16 * v42 - 16];
    v49 = *(_QWORD *)v48;
    LOBYTE(v48) = v48[8];
  }
  else
  {
    v40 = sub_3007260((__int64)&v80);
    v48 = v47;
    v88 = v40;
    v49 = v40;
    v89 = v48;
  }
  if ( v44 != v49 || (_BYTE)v48 != v46 )
  {
    v21 = sub_37A59E0((__int64)a1, a2, v7, v49, v40, v23);
    goto LABEL_38;
  }
LABEL_8:
  v20 = *(_DWORD *)(a2 + 24);
  if ( v20 != 214 )
  {
    if ( v20 == 215 )
    {
      v21 = sub_33FAF80(a1[1], 223, (__int64)&v76, v78, v79, v9, v7);
      goto LABEL_38;
    }
    if ( v20 == 213 )
    {
      v21 = sub_33FAF80(a1[1], 224, (__int64)&v76, v78, v79, v9, v7);
      goto LABEL_38;
    }
LABEL_73:
    BUG();
  }
  v21 = sub_33FAF80(a1[1], 225, (__int64)&v76, v78, v79, v9, v7);
LABEL_38:
  if ( v76 )
    sub_B91220((__int64)&v76, v76);
  return v21;
}
