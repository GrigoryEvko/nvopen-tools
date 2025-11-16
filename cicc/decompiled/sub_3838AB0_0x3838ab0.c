// Function: sub_3838AB0
// Address: 0x3838ab0
//
__int64 __fastcall sub_3838AB0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v9; // r8d
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  unsigned __int16 v17; // dx
  __int64 v18; // r12
  __int64 v19; // r15
  bool v20; // al
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned __int16 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r9
  int v30; // r9d
  __int64 v31; // r12
  unsigned int v33; // r8d
  unsigned __int16 v34; // r9
  __int64 v35; // r10
  __int64 v36; // r10
  int v37; // r9d
  const __m128i *v38; // rax
  unsigned __int8 *v39; // rax
  unsigned int v40; // edx
  __int64 v41; // r9
  __int64 v42; // r13
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 v45; // rdx
  bool v46; // zf
  __int64 v47; // r9
  __int64 v48; // rax
  _QWORD *v49; // r12
  int v50; // r9d
  __int128 v51; // rax
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // r12
  __int64 v55; // r13
  __int128 v56; // rax
  _QWORD *v57; // r14
  __int64 v58; // r9
  __int128 v59; // rax
  __int64 v60; // r9
  unsigned __int16 v61; // ax
  __int64 v62; // rdx
  __int64 v63; // r8
  __int128 v64; // [rsp-40h] [rbp-120h]
  __int128 v65; // [rsp-30h] [rbp-110h]
  __int128 v66; // [rsp-20h] [rbp-100h]
  __int128 v67; // [rsp-20h] [rbp-100h]
  __int128 v68; // [rsp-20h] [rbp-100h]
  __int128 v69; // [rsp-10h] [rbp-F0h]
  __int128 v70; // [rsp-10h] [rbp-F0h]
  __int128 v71; // [rsp+0h] [rbp-E0h]
  __int128 v72; // [rsp+10h] [rbp-D0h]
  unsigned int v73; // [rsp+20h] [rbp-C0h]
  __int128 v74; // [rsp+20h] [rbp-C0h]
  __int128 v75; // [rsp+20h] [rbp-C0h]
  unsigned int v76; // [rsp+30h] [rbp-B0h]
  __m128i v77; // [rsp+30h] [rbp-B0h]
  __int64 v78; // [rsp+50h] [rbp-90h] BYREF
  __int64 v79; // [rsp+58h] [rbp-88h]
  unsigned int v80; // [rsp+60h] [rbp-80h] BYREF
  __int64 v81; // [rsp+68h] [rbp-78h]
  __int64 v82; // [rsp+70h] [rbp-70h] BYREF
  int v83; // [rsp+78h] [rbp-68h]
  unsigned __int16 v84; // [rsp+80h] [rbp-60h] BYREF
  __int64 v85; // [rsp+88h] [rbp-58h]
  __int64 v86; // [rsp+90h] [rbp-50h] BYREF
  __int64 v87; // [rsp+98h] [rbp-48h]
  __int64 v88; // [rsp+A0h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 48);
  v5 = *a1;
  v6 = *(_QWORD *)(v4 + 8);
  LOWORD(v78) = *(_WORD *)v4;
  v7 = a1[1];
  v79 = v6;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  if ( v8 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v86, v5, *(_QWORD *)(v7 + 64), v78, v79);
    LOWORD(v80) = v87;
    v81 = v88;
  }
  else
  {
    v80 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v8)(v5, *(_QWORD *)(v7 + 64), (unsigned int)v78);
    v81 = v53;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v82 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v82, v10, 1);
  v83 = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v78 )
  {
    if ( (unsigned __int16)(v78 - 17) <= 0xD3u )
      goto LABEL_7;
  }
  else if ( sub_30070B0((__int64)&v78) )
  {
    goto LABEL_7;
  }
  if ( (_WORD)v80 )
  {
    if ( *(_QWORD *)(*a1 + 8LL * (unsigned __int16)v80 + 112) )
    {
      v10 = 199;
      if ( !(unsigned __int8)sub_38138F0(*a1, 0xC7u, v80, 0, v9) )
      {
        v10 = 204;
        if ( !(unsigned __int8)sub_38138F0(v35, 0xCCu, v34, 0, v33) )
        {
          v10 = a2;
          if ( sub_3457D60(v36, a2, a1[1], a3) )
          {
            v31 = (__int64)sub_33FAF80(a1[1], 215, (__int64)&v82, v80, v81, v37, a3);
            goto LABEL_17;
          }
        }
      }
    }
  }
LABEL_7:
  v11 = *(_DWORD *)(a2 + 24);
  v76 = v11;
  if ( v11 != 199 && v11 != 416 )
  {
    if ( v11 != 204 && v11 != 417 )
      BUG();
    v12 = *(__int64 **)(a2 + 40);
    v13 = *v12;
    v14 = sub_37AE0F0((__int64)a1, *v12, v12[1]);
    v73 = v15;
    v16 = v15;
    v17 = v80;
    v18 = v14;
    v19 = v14;
    if ( (_WORD)v80 )
    {
      if ( (unsigned __int16)(v80 - 17) <= 0xD3u )
      {
        v17 = word_4456580[(unsigned __int16)v80 - 1];
        v85 = 0;
        v84 = v17;
        if ( !v17 )
          goto LABEL_14;
        goto LABEL_29;
      }
    }
    else
    {
      v20 = sub_30070B0((__int64)&v80);
      v17 = 0;
      if ( v20 )
      {
        v61 = sub_3009970((__int64)&v80, v13, 0, v21, v22);
        v63 = v62;
        v17 = v61;
        v23 = v63;
LABEL_13:
        v84 = v17;
        v85 = v23;
        if ( !v17 )
        {
LABEL_14:
          v86 = sub_3007260((__int64)&v84);
          v24 = v86;
          v87 = v25;
LABEL_15:
          v26 = (unsigned int)v24 - (unsigned int)sub_32844A0((unsigned __int16 *)&v78, v24);
          v27 = (unsigned __int16 *)(*(_QWORD *)(v19 + 48) + 16LL * v73);
          *(_QWORD *)&v74 = sub_3400E40(a1[1], v26, *v27, *((_QWORD *)v27 + 1), (__int64)&v82, a3);
          *((_QWORD *)&v74 + 1) = v28;
          if ( sub_33CB110(*(_DWORD *)(a2 + 24)) )
          {
            v38 = *(const __m128i **)(a2 + 40);
            *(_QWORD *)&v71 = v38[2].m128i_i64[1];
            v72 = (__int128)_mm_loadu_si128(v38 + 5);
            *((_QWORD *)&v67 + 1) = v38[3].m128i_i64[0];
            *(_QWORD *)&v67 = v71;
            *((_QWORD *)&v64 + 1) = v16;
            *(_QWORD *)&v64 = v18;
            *((_QWORD *)&v71 + 1) = *((_QWORD *)&v67 + 1);
            v39 = sub_33FC130((_QWORD *)a1[1], 402, (__int64)&v82, v80, v81, v29, v64, v74, v67, v72);
            *((_QWORD *)&v65 + 1) = v40 | v16 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v65 = v39;
            v31 = sub_340F900((_QWORD *)a1[1], v76, (__int64)&v82, v80, v81, v41, v65, v71, v72);
          }
          else
          {
            *((_QWORD *)&v66 + 1) = v16;
            *(_QWORD *)&v66 = v18;
            sub_3406EB0((_QWORD *)a1[1], 0xBEu, (__int64)&v82, v80, v81, v29, v66, v74);
            v31 = (__int64)sub_33FAF80(a1[1], v76, (__int64)&v82, v80, v81, v30, a3);
          }
          goto LABEL_17;
        }
LABEL_29:
        if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
          BUG();
        v24 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
        goto LABEL_15;
      }
    }
    v23 = v81;
    goto LABEL_13;
  }
  v42 = a1[1];
  v43 = sub_32844A0((unsigned __int16 *)&v80, v10);
  v44 = sub_32844A0((unsigned __int16 *)&v78, v10);
  *(_QWORD *)&v75 = sub_3400BD0(v42, v43 - v44, (__int64)&v82, v80, v81, 0, a3, 0);
  *((_QWORD *)&v75 + 1) = v45;
  v46 = !sub_33CB110(*(_DWORD *)(a2 + 24));
  v48 = *(_QWORD *)(a2 + 40);
  if ( v46 )
  {
    sub_37AF270((__int64)a1, *(_QWORD *)v48, *(_QWORD *)(v48 + 8), a3);
    v49 = (_QWORD *)a1[1];
    *(_QWORD *)&v51 = sub_33FAF80((__int64)v49, *(unsigned int *)(a2 + 24), (__int64)&v82, v80, v81, v50, a3);
    v31 = (__int64)sub_3406EB0(v49, 0x39u, (__int64)&v82, v80, v81, v52, v51, v75);
  }
  else
  {
    v54 = *(_QWORD *)(v48 + 80);
    v55 = *(_QWORD *)(v48 + 88);
    v77 = _mm_loadu_si128((const __m128i *)(v48 + 40));
    *((_QWORD *)&v69 + 1) = v55;
    *(_QWORD *)&v69 = v54;
    *(_QWORD *)&v56 = sub_3838540(
                        (__int64)a1,
                        *(_QWORD *)v48,
                        *(_QWORD *)(v48 + 8),
                        v77.m128i_i64[0],
                        v77.m128i_i64[1],
                        a3,
                        v47,
                        v69);
    v57 = (_QWORD *)a1[1];
    *((_QWORD *)&v68 + 1) = v55;
    *(_QWORD *)&v68 = v54;
    *(_QWORD *)&v59 = sub_340F900(v57, *(_DWORD *)(a2 + 24), (__int64)&v82, v80, v81, v58, v56, *(_OWORD *)&v77, v68);
    *((_QWORD *)&v70 + 1) = v55;
    *(_QWORD *)&v70 = v54;
    v31 = (__int64)sub_33FC130(v57, 404, (__int64)&v82, v80, v81, v60, v59, v75, *(_OWORD *)&v77, v70);
  }
LABEL_17:
  if ( v82 )
    sub_B91220((__int64)&v82, v82);
  return v31;
}
