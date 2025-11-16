// Function: sub_326E5E0
// Address: 0x326e5e0
//
__int64 __fastcall sub_326E5E0(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // rsi
  const __m128i *v5; // rax
  __int64 v6; // r12
  __int64 v7; // r15
  __m128i v8; // xmm1
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // r8
  __int16 v18; // dx
  __int64 v19; // rax
  int v20; // r14d
  __int64 i; // r12
  _QWORD *v22; // rax
  __int128 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r9
  unsigned __int64 v27; // rdx
  __int64 *v28; // rax
  int v30; // eax
  __int64 v31; // rdx
  unsigned __int64 v32; // r14
  __int64 v33; // r12
  __int128 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int64 *v39; // rax
  unsigned int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int128 v44; // [rsp-20h] [rbp-220h]
  __int128 v45; // [rsp-10h] [rbp-210h]
  __int64 v46; // [rsp+0h] [rbp-200h]
  __int64 v47; // [rsp+10h] [rbp-1F0h]
  __int64 v48; // [rsp+10h] [rbp-1F0h]
  __int64 v49; // [rsp+18h] [rbp-1E8h]
  unsigned int v50; // [rsp+24h] [rbp-1DCh]
  __int64 v51; // [rsp+30h] [rbp-1D0h]
  __int16 v52; // [rsp+3Ah] [rbp-1C6h]
  unsigned int v53; // [rsp+3Ch] [rbp-1C4h]
  __int64 v54; // [rsp+40h] [rbp-1C0h]
  __int64 v55; // [rsp+40h] [rbp-1C0h]
  __int64 v56; // [rsp+48h] [rbp-1B8h]
  __int64 v57; // [rsp+50h] [rbp-1B0h]
  unsigned __int32 v58; // [rsp+58h] [rbp-1A8h]
  int v59; // [rsp+5Ch] [rbp-1A4h]
  __m128i v60; // [rsp+60h] [rbp-1A0h]
  _QWORD *v61; // [rsp+60h] [rbp-1A0h]
  __int64 v62; // [rsp+70h] [rbp-190h]
  __int64 v63; // [rsp+70h] [rbp-190h]
  __int64 v64; // [rsp+78h] [rbp-188h]
  __int64 v65; // [rsp+80h] [rbp-180h] BYREF
  int v66; // [rsp+88h] [rbp-178h]
  __int64 v67; // [rsp+90h] [rbp-170h] BYREF
  __int64 v68; // [rsp+98h] [rbp-168h]
  unsigned __int64 v69; // [rsp+A0h] [rbp-160h] BYREF
  unsigned int v70; // [rsp+A8h] [rbp-158h]
  __int64 v71; // [rsp+B0h] [rbp-150h] BYREF
  int v72; // [rsp+B8h] [rbp-148h]
  _BYTE *v73; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v74; // [rsp+C8h] [rbp-138h]
  _BYTE v75[304]; // [rsp+D0h] [rbp-130h] BYREF

  v3 = a1;
  v4 = *(_QWORD *)(a2 + 80);
  v65 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v65, v4, 1);
  v66 = *(_DWORD *)(a2 + 72);
  v5 = *(const __m128i **)(a2 + 40);
  v6 = v5->m128i_i64[0];
  v7 = v5[2].m128i_i64[1];
  v8 = _mm_loadu_si128(v5 + 5);
  v60 = _mm_loadu_si128(v5);
  v9 = v5[3].m128i_i64[0];
  v57 = v5[5].m128i_i64[0];
  v10 = v7;
  v58 = v5[5].m128i_u32[2];
  v54 = v5->m128i_u32[2];
  v11 = *(_QWORD *)(v5->m128i_i64[0] + 48) + 16 * v54;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v70 = 1;
  v68 = v13;
  LODWORD(v13) = *(_DWORD *)(v57 + 24);
  LOWORD(v67) = v12;
  v59 = v13;
  v69 = 0;
  if ( (unsigned __int8)sub_33D1410(v7, &v69) )
  {
    if ( (unsigned __int8)sub_3449B70(a1[1], v7, v9) )
    {
      v15 = v60.m128i_i64[0];
      goto LABEL_21;
    }
    goto LABEL_20;
  }
  if ( *(_DWORD *)(v6 + 24) == 51 || *(_DWORD *)(v7 + 24) == 51 )
  {
LABEL_20:
    v15 = v8.m128i_i64[0];
    goto LABEL_21;
  }
  v14 = v7;
  v15 = 0;
  if ( !(unsigned __int8)sub_33CA6D0(v14) )
    goto LABEL_21;
  v18 = v67;
  v73 = v75;
  v74 = 0x1000000000LL;
  if ( (_WORD)v67 )
  {
    v51 = 0;
    v19 = (unsigned __int16)v67 - 1;
    v20 = (unsigned __int16)word_4456580[v19];
    v52 = word_4456580[v19];
  }
  else
  {
    v30 = sub_3009970((__int64)&v67, (__int64)&v69, 0, v16, v17);
    v51 = v31;
    v18 = v67;
    HIWORD(v20) = HIWORD(v30);
    v52 = v30;
    if ( !(_WORD)v67 )
    {
      if ( !sub_3007100((__int64)&v67) )
        goto LABEL_30;
      goto LABEL_44;
    }
  }
  if ( (unsigned __int16)(v18 - 176) <= 0x34u )
  {
LABEL_44:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v67 )
    {
      if ( (unsigned __int16)(v67 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_10;
    }
LABEL_30:
    v53 = sub_3007130((__int64)&v67, (__int64)&v69);
    goto LABEL_11;
  }
LABEL_10:
  v53 = word_4456340[(unsigned __int16)v67 - 1];
LABEL_11:
  if ( v53 )
  {
    v46 = v6;
    v50 = 0;
    for ( i = 0; i != v53; ++i )
    {
      v22 = (_QWORD *)(*(_QWORD *)(v10 + 40) + 40 * i);
      if ( *(_DWORD *)(*v22 + 24LL) != 51 && (unsigned __int8)sub_3449B70(v3[1], *v22, v22[1]) )
      {
        v47 = *v3;
        *(_QWORD *)&v23 = sub_3400EE0(*v3, i, &v65, 0, v17);
        LOWORD(v20) = v52;
        v60.m128i_i64[1] = v54 | v60.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v17 = sub_3406EB0(v47, 158, (unsigned int)&v65, v20, v51, DWORD2(v23), __PAIR128__(v60.m128i_u64[1], v46), v23);
        v24 = (unsigned int)v74;
        v26 = v25;
        v27 = (unsigned int)v74 + 1LL;
        if ( v27 > HIDWORD(v74) )
        {
          v48 = v17;
          v49 = v26;
          sub_C8D5F0((__int64)&v73, v75, v27, 0x10u, v17, v26);
          v24 = (unsigned int)v74;
          v17 = v48;
          v26 = v49;
        }
        v28 = (__int64 *)&v73[16 * v24];
        ++v50;
        *v28 = v17;
        v28[1] = v26;
        LODWORD(v74) = v74 + 1;
      }
    }
    if ( v50 >= v53 )
    {
      v40 = v74;
    }
    else
    {
      v61 = v3;
      WORD1(v3) = HIWORD(v20);
      v32 = v8.m128i_u64[1];
      v33 = v50;
      do
      {
        v41 = *v61;
        if ( v59 == 51 )
        {
          LOWORD(v3) = v52;
          v71 = 0;
          v72 = 0;
          v42 = sub_33F17F0(v41, 51, &v71, (unsigned int)v3, v51);
          if ( v71 )
          {
            v55 = v42;
            v56 = v43;
            sub_B91220((__int64)&v71, v71);
            v42 = v55;
            v43 = v56;
          }
          v17 = v42;
          v36 = v43;
        }
        else
        {
          v62 = *v61;
          *(_QWORD *)&v34 = sub_3400EE0(v41, v33, &v65, 0, v17);
          LOWORD(v3) = v52;
          v32 = v58 | v32 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v44 + 1) = v32;
          *(_QWORD *)&v44 = v57;
          v17 = sub_3406EB0(v62, 158, (unsigned int)&v65, (_DWORD)v3, v51, DWORD2(v34), v44, v34);
          v36 = v35;
        }
        v37 = (unsigned int)v74;
        v38 = (unsigned int)v74 + 1LL;
        if ( v38 > HIDWORD(v74) )
        {
          v63 = v17;
          v64 = v36;
          sub_C8D5F0((__int64)&v73, v75, v38, 0x10u, v17, v36);
          v37 = (unsigned int)v74;
          v36 = v64;
          v17 = v63;
        }
        v39 = (__int64 *)&v73[16 * v37];
        ++v33;
        *v39 = v17;
        v39[1] = v36;
        v40 = v74 + 1;
        LODWORD(v74) = v74 + 1;
      }
      while ( v53 > (unsigned int)v33 );
      v3 = v61;
    }
  }
  else
  {
    v40 = v74;
  }
  *((_QWORD *)&v45 + 1) = v40;
  *(_QWORD *)&v45 = v73;
  v15 = sub_33FC220(*v3, 156, (unsigned int)&v65, v67, v68, *v3, v45);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
LABEL_21:
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  if ( v65 )
    sub_B91220((__int64)&v65, v65);
  return v15;
}
