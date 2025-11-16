// Function: sub_383B570
// Address: 0x383b570
//
__int64 __fastcall sub_383B570(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int64 v5; // rdx
  int v6; // eax
  bool v7; // r14
  unsigned int v8; // edx
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // r9
  unsigned int v13; // edx
  __int64 v14; // rax
  unsigned __int16 v15; // dx
  unsigned __int16 *v16; // rdx
  int v17; // eax
  __int64 v18; // r15
  __int64 v19; // r15
  unsigned __int16 v20; // dx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // rdi
  unsigned int v25; // r15d
  __int128 v26; // rax
  unsigned __int64 v27; // r13
  unsigned __int8 *v28; // r12
  unsigned int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // r12
  _QWORD *v34; // rbx
  __int128 v35; // rax
  __int64 v36; // r12
  unsigned int v38; // edx
  __int64 v39; // rax
  unsigned int v40; // edx
  bool v41; // al
  __int64 v42; // rcx
  __int64 v43; // r8
  unsigned __int16 v44; // ax
  __int64 v45; // rdx
  __int64 v46; // r8
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int128 v52; // [rsp-30h] [rbp-130h]
  __int128 v53; // [rsp-30h] [rbp-130h]
  __int128 v54; // [rsp-20h] [rbp-120h]
  __int128 v55; // [rsp-20h] [rbp-120h]
  _QWORD *v56; // [rsp+8h] [rbp-F8h]
  char v57; // [rsp+17h] [rbp-E9h]
  unsigned int v58; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v59; // [rsp+20h] [rbp-E0h]
  __int128 v61; // [rsp+30h] [rbp-D0h]
  __int64 v62; // [rsp+70h] [rbp-90h] BYREF
  int v63; // [rsp+78h] [rbp-88h]
  unsigned __int16 v64; // [rsp+80h] [rbp-80h] BYREF
  __int64 v65; // [rsp+88h] [rbp-78h]
  unsigned int v66; // [rsp+90h] [rbp-70h] BYREF
  __int64 v67; // [rsp+98h] [rbp-68h]
  unsigned __int16 v68; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v69; // [rsp+A8h] [rbp-58h]
  __int64 v70; // [rsp+B0h] [rbp-50h]
  __int64 v71; // [rsp+B8h] [rbp-48h]
  __int64 v72; // [rsp+C0h] [rbp-40h] BYREF
  __int64 v73; // [rsp+C8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v62 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v62, v4, 1);
  v63 = *(_DWORD *)(a2 + 72);
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_DWORD *)(a2 + 24);
  if ( v6 == 88 )
  {
    v7 = 0;
  }
  else
  {
    if ( v6 != 90 )
    {
      v7 = v6 == 91;
      v59 = sub_37AF270(a1, *(_QWORD *)v5, *(_QWORD *)(v5 + 8), a3);
      v58 = v8;
      v9 = v8;
      v10 = *(_QWORD *)(a2 + 40);
      v11 = *(_QWORD *)(v10 + 40);
      v57 = 0;
      *(_QWORD *)&v61 = sub_37AF270(a1, v11, *(_QWORD *)(v10 + 48), a3);
      *((_QWORD *)&v61 + 1) = v13;
      goto LABEL_6;
    }
    v7 = 1;
  }
  v59 = sub_383B380(a1, *(_QWORD *)v5, *(_QWORD *)(v5 + 8));
  v58 = v38;
  v9 = v38;
  v39 = *(_QWORD *)(a2 + 40);
  v11 = *(_QWORD *)(v39 + 40);
  v57 = 1;
  *(_QWORD *)&v61 = sub_383B380(a1, v11, *(_QWORD *)(v39 + 48));
  *((_QWORD *)&v61 + 1) = v40;
LABEL_6:
  v14 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v15 = *(_WORD *)v14;
  v65 = *(_QWORD *)(v14 + 8);
  v64 = v15;
  v16 = (unsigned __int16 *)(*((_QWORD *)v59 + 6) + 16LL * v58);
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  LOWORD(v66) = v17;
  v67 = v18;
  if ( (_WORD)v17 )
  {
    if ( (unsigned __int16)(v17 - 17) > 0xD3u )
    {
      LOWORD(v72) = v17;
      v73 = v18;
      goto LABEL_9;
    }
    LOWORD(v17) = word_4456580[v17 - 1];
    v51 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v66) )
    {
      v73 = v18;
      LOWORD(v72) = 0;
LABEL_32:
      v70 = sub_3007260((__int64)&v72);
      LODWORD(v19) = v70;
      v71 = v50;
      goto LABEL_12;
    }
    LOWORD(v17) = sub_3009970((__int64)&v66, v11, v47, v48, v49);
  }
  LOWORD(v72) = v17;
  v73 = v51;
  if ( !(_WORD)v17 )
    goto LABEL_32;
LABEL_9:
  if ( (_WORD)v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
LABEL_39:
    BUG();
  v19 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v17 - 16];
LABEL_12:
  v20 = v64;
  if ( !v64 )
  {
    v41 = sub_30070B0((__int64)&v64);
    v20 = 0;
    if ( v41 )
    {
      v44 = sub_3009970((__int64)&v64, v11, 0, v42, v43);
      v46 = v45;
      v20 = v44;
      v68 = v44;
      v69 = v46;
      if ( !v44 )
        goto LABEL_16;
      goto LABEL_27;
    }
    goto LABEL_14;
  }
  if ( (unsigned __int16)(v64 - 17) > 0xD3u )
  {
LABEL_14:
    v21 = v65;
    goto LABEL_15;
  }
  v20 = word_4456580[v64 - 1];
  v21 = 0;
LABEL_15:
  v68 = v20;
  v69 = v21;
  if ( !v20 )
  {
LABEL_16:
    v22 = sub_3007260((__int64)&v68);
    v72 = v22;
    v73 = v23;
    goto LABEL_17;
  }
LABEL_27:
  if ( v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
    goto LABEL_39;
  v22 = *(_QWORD *)&byte_444C4A0[16 * v20 - 16];
LABEL_17:
  v24 = *(_QWORD **)(a1 + 8);
  if ( v7 )
  {
    v25 = v19 - v22;
    v56 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v26 = sub_3400E40((__int64)v24, v25, v66, v67, (__int64)&v62, a3);
    v27 = v9 & 0xFFFFFFFF00000000LL | v58;
    *((_QWORD *)&v54 + 1) = v27;
    *(_QWORD *)&v54 = v59;
    v28 = sub_3406EB0(v56, 0xBEu, (__int64)&v62, v66, v67, 0xFFFFFFFF00000000LL, v54, v26);
    *((_QWORD *)&v52 + 1) = v29 | v27 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v52 = v28;
    v30 = sub_340F900(
            *(_QWORD **)(a1 + 8),
            *(_DWORD *)(a2 + 24),
            (__int64)&v62,
            v66,
            v67,
            0xFFFFFFFF00000000LL,
            v52,
            v61,
            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
    v32 = v31;
    v33 = v30;
    v34 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v35 = sub_3400E40((__int64)v34, v25, v66, v67, (__int64)&v62, a3);
    *((_QWORD *)&v55 + 1) = v32;
    *(_QWORD *)&v55 = v33;
    v36 = (__int64)sub_3406EB0(
                     v34,
                     (unsigned int)(v57 == 0) + 191,
                     (__int64)&v62,
                     v66,
                     v67,
                     (unsigned int)(v57 == 0) + 191,
                     v55,
                     v35);
  }
  else
  {
    *((_QWORD *)&v53 + 1) = v9 & 0xFFFFFFFF00000000LL | v58;
    *(_QWORD *)&v53 = v59;
    v36 = sub_340F900(
            v24,
            *(_DWORD *)(a2 + 24),
            (__int64)&v62,
            v66,
            v67,
            v12,
            v53,
            v61,
            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  }
  if ( v62 )
    sub_B91220((__int64)&v62, v62);
  return v36;
}
