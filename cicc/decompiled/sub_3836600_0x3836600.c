// Function: sub_3836600
// Address: 0x3836600
//
__int64 __fastcall sub_3836600(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rsi
  unsigned __int16 *v9; // rdx
  unsigned int v10; // ecx
  int v11; // eax
  __int64 v12; // r12
  unsigned int v13; // ecx
  __int64 v14; // rax
  int *v15; // r12
  __int64 v16; // r8
  __int64 v17; // rsi
  int v18; // ecx
  __int64 v19; // rax
  int v20; // r9d
  __int64 v21; // rsi
  unsigned __int16 *v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rcx
  __int64 v25; // rcx
  unsigned __int16 v26; // r14
  __int64 v27; // rax
  unsigned int v28; // eax
  __int64 v29; // r14
  __int128 v30; // rax
  __int64 v31; // r9
  int v32; // r9d
  unsigned __int8 *v33; // r8
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r9
  unsigned __int64 v37; // rdx
  unsigned __int8 **v38; // rax
  _BYTE *v39; // r9
  __int64 v40; // rsi
  __int64 v41; // r12
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  _QWORD *v46; // rdi
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // r13
  __int64 v49; // r14
  __int64 v50; // rax
  __int128 v51; // rax
  __int64 v52; // r9
  unsigned int v53; // edx
  int v54; // esi
  __int128 v55; // xmm0
  __int64 v56; // rax
  unsigned __int16 v57; // dx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  int v62; // eax
  __int64 v63; // rdx
  int v64; // r10d
  __int128 v65; // [rsp-30h] [rbp-1A0h]
  __int128 v66; // [rsp-20h] [rbp-190h]
  __int128 v67; // [rsp-10h] [rbp-180h]
  unsigned int v69; // [rsp+28h] [rbp-148h]
  unsigned int i; // [rsp+2Ch] [rbp-144h]
  __int64 v71; // [rsp+30h] [rbp-140h]
  unsigned int v72; // [rsp+38h] [rbp-138h]
  __int16 v73; // [rsp+3Ah] [rbp-136h]
  __int64 v74; // [rsp+40h] [rbp-130h]
  __int64 v75; // [rsp+48h] [rbp-128h]
  __int64 v76; // [rsp+58h] [rbp-118h]
  _QWORD *v77; // [rsp+58h] [rbp-118h]
  _QWORD *v78; // [rsp+60h] [rbp-110h]
  unsigned __int8 *v79; // [rsp+60h] [rbp-110h]
  __int64 v80; // [rsp+68h] [rbp-108h]
  __int64 v81; // [rsp+80h] [rbp-F0h] BYREF
  int v82; // [rsp+88h] [rbp-E8h]
  __int64 v83; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v84; // [rsp+98h] [rbp-D8h]
  __int64 v85; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v86; // [rsp+A8h] [rbp-C8h]
  _BYTE *v87; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v88; // [rsp+B8h] [rbp-B8h]
  _BYTE v89[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v81 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v81, v8, 1);
  v9 = *(unsigned __int16 **)(a2 + 48);
  v10 = *(_DWORD *)(a2 + 64);
  v82 = *(_DWORD *)(a2 + 72);
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v69 = v10;
  LOWORD(v83) = v11;
  v84 = v12;
  if ( !(_WORD)v11 )
  {
    if ( !sub_3007100((__int64)&v83) )
    {
      v88 = v12;
      LOWORD(v87) = 0;
      v11 = sub_3009970((__int64)&v87, a2, v43, v44, v45);
      v73 = HIWORD(v11);
      v71 = v63;
      goto LABEL_6;
    }
LABEL_30:
    v46 = *(_QWORD **)(a1 + 8);
    v87 = 0;
    LODWORD(v88) = 0;
    v41 = (__int64)sub_33F17F0(v46, 51, (__int64)&v87, v83, v84);
    v48 = v47;
    if ( v87 )
      sub_B91220((__int64)&v87, (__int64)v87);
    v49 = 0;
    if ( v69 )
    {
      do
      {
        v54 = v49;
        v55 = (__int128)_mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40 * v49));
        v56 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40 * v49) + 48LL)
            + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 40 * v49 + 8);
        v57 = *(_WORD *)v56;
        v58 = *(_QWORD *)(v56 + 8);
        LOWORD(v87) = v57;
        v88 = v58;
        if ( v57 )
        {
          LODWORD(v50) = word_4456340[v57 - 1];
        }
        else
        {
          v50 = sub_3007240((__int64)&v87);
          v54 = v49;
          v85 = v50;
        }
        ++v49;
        v77 = *(_QWORD **)(a1 + 8);
        *(_QWORD *)&v51 = sub_3400D50((__int64)v77, (unsigned int)(v50 * v54), (__int64)&v81, 0, (__m128i)v55);
        *((_QWORD *)&v65 + 1) = v48;
        *(_QWORD *)&v65 = v41;
        v41 = sub_340F900(v77, 0xA0u, (__int64)&v81, v83, v84, v52, v65, v55, v51);
        v48 = v53 | v48 & 0xFFFFFFFF00000000LL;
      }
      while ( v49 != v69 );
    }
    goto LABEL_26;
  }
  if ( (unsigned __int16)(v11 - 176) <= 0x34u )
    goto LABEL_30;
  v71 = 0;
  LOWORD(v11) = word_4456580[v11 - 1];
LABEL_6:
  HIWORD(v13) = v73;
  LOWORD(v13) = v11;
  v87 = v89;
  v72 = v13;
  v88 = 0x800000000LL;
  if ( v69 > 8 )
  {
    sub_C8D5F0((__int64)&v87, v89, v69, 0x10u, a6, a7);
  }
  else if ( !v69 )
  {
    v39 = v89;
    v40 = 0;
    goto LABEL_24;
  }
  for ( i = 0; i != v69; ++i )
  {
    v14 = *(_QWORD *)(a2 + 40) + 40LL * i;
    LODWORD(v85) = sub_375D5B0(a1, *(_QWORD *)v14, *(_QWORD *)(v14 + 8));
    v15 = sub_3805BC0(a1 + 712, (int *)&v85);
    sub_37593F0(a1, v15);
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v17 = a1 + 520;
      v18 = 7;
    }
    else
    {
      v59 = *(unsigned int *)(a1 + 528);
      v17 = *(_QWORD *)(a1 + 520);
      if ( !(_DWORD)v59 )
        goto LABEL_45;
      v18 = v59 - 1;
    }
    v16 = v18 & (unsigned int)(37 * *v15);
    v19 = v17 + 24 * v16;
    v20 = *(_DWORD *)v19;
    if ( *v15 == *(_DWORD *)v19 )
      goto LABEL_12;
    v62 = 1;
    while ( v20 != -1 )
    {
      v64 = v62 + 1;
      v16 = v18 & (unsigned int)(v62 + v16);
      v19 = v17 + 24LL * (unsigned int)v16;
      v20 = *(_DWORD *)v19;
      if ( *v15 == *(_DWORD *)v19 )
        goto LABEL_12;
      v62 = v64;
    }
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v61 = 192;
      goto LABEL_46;
    }
    v59 = *(unsigned int *)(a1 + 528);
LABEL_45:
    v61 = 24 * v59;
LABEL_46:
    v19 = v17 + v61;
LABEL_12:
    v21 = *(_QWORD *)(v19 + 8);
    v22 = *(unsigned __int16 **)(v21 + 48);
    v23 = *(unsigned int *)(v19 + 16);
    v24 = *((_QWORD *)v22 + 1);
    LODWORD(v19) = *v22;
    v86 = v24;
    LOWORD(v85) = v19;
    if ( (_WORD)v19 )
    {
      v74 = 0;
      LOWORD(v19) = word_4456580[(int)v19 - 1];
    }
    else
    {
      v19 = sub_3009970((__int64)&v85, v21, (__int64)v22, v24, v16);
      v74 = v60;
      v76 = v19;
      v22 = *(unsigned __int16 **)(v21 + 48);
    }
    v25 = v76;
    v26 = *v22;
    LOWORD(v25) = v19;
    v27 = *((_QWORD *)v22 + 1);
    LOWORD(v85) = v26;
    v76 = v25;
    v86 = v27;
    if ( v26 )
    {
      if ( (unsigned __int16)(v26 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      }
      v28 = word_4456340[v26 - 1];
    }
    else
    {
      if ( sub_3007100((__int64)&v85) )
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
      v28 = sub_3007130((__int64)&v85, v21);
    }
    v29 = 0;
    v75 = v28;
    if ( v28 )
    {
      do
      {
        v78 = *(_QWORD **)(a1 + 8);
        *(_QWORD *)&v30 = sub_3400EE0((__int64)v78, v29, (__int64)&v81, 0, a3);
        *((_QWORD *)&v66 + 1) = v23;
        *(_QWORD *)&v66 = v21;
        sub_3406EB0(v78, 0x9Eu, (__int64)&v81, (unsigned int)v76, v74, v31, v66, v30);
        v33 = sub_33FAF80(*(_QWORD *)(a1 + 8), 216, (__int64)&v81, v72, v71, v32, a3);
        v34 = (unsigned int)v88;
        v36 = v35;
        v37 = (unsigned int)v88 + 1LL;
        if ( v37 > HIDWORD(v88) )
        {
          v79 = v33;
          v80 = v36;
          sub_C8D5F0((__int64)&v87, v89, v37, 0x10u, (__int64)v33, v36);
          v34 = (unsigned int)v88;
          v36 = v80;
          v33 = v79;
        }
        v38 = (unsigned __int8 **)&v87[16 * v34];
        ++v29;
        *v38 = v33;
        v38[1] = (unsigned __int8 *)v36;
        LODWORD(v88) = v88 + 1;
      }
      while ( v75 != v29 );
    }
  }
  v39 = v87;
  v40 = (unsigned int)v88;
LABEL_24:
  *((_QWORD *)&v67 + 1) = v40;
  *(_QWORD *)&v67 = v39;
  v41 = (__int64)sub_33FC220(
                   *(_QWORD **)(a1 + 8),
                   156,
                   (__int64)&v81,
                   **(unsigned __int16 **)(a2 + 48),
                   *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                   (__int64)v39,
                   v67);
  if ( v87 != v89 )
    _libc_free((unsigned __int64)v87);
LABEL_26:
  if ( v81 )
    sub_B91220((__int64)&v81, v81);
  return v41;
}
