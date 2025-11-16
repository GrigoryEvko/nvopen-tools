// Function: sub_37A38B0
// Address: 0x37a38b0
//
unsigned __int8 *__fastcall sub_37A38B0(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // rbx
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  int v9; // edx
  unsigned __int64 v10; // rdx
  const __m128i *v11; // r13
  __int64 v12; // r12
  __m128i v13; // xmm0
  __int64 v14; // rax
  unsigned __int16 v15; // dx
  unsigned __int8 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r13
  unsigned __int8 *v21; // r12
  __int64 v22; // rdx
  __int128 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  _QWORD *v27; // r14
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  _QWORD *v34; // r13
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rsi
  __int64 v38; // rsi
  __int64 v39; // r12
  __int64 v41; // rsi
  __int64 v42; // r14
  unsigned __int8 *v43; // rax
  __int64 v44; // rdx
  int v45; // eax
  unsigned int v46; // esi
  __int64 v47; // rdx
  __int64 v48; // r13
  __int64 v49; // rax
  unsigned int v50; // edx
  unsigned __int8 *v51; // rax
  __int64 v52; // rdx
  unsigned __int32 v53; // edx
  __int128 v54; // [rsp-30h] [rbp-100h]
  __int128 v55; // [rsp-20h] [rbp-F0h]
  __int128 v56; // [rsp-10h] [rbp-E0h]
  __int128 v57; // [rsp-10h] [rbp-E0h]
  __int64 v58; // [rsp+0h] [rbp-D0h]
  __int64 v59; // [rsp+0h] [rbp-D0h]
  unsigned int v60; // [rsp+0h] [rbp-D0h]
  __int16 v61; // [rsp+2h] [rbp-CEh]
  __int64 v62; // [rsp+8h] [rbp-C8h]
  __int64 v63; // [rsp+8h] [rbp-C8h]
  unsigned __int32 v64; // [rsp+10h] [rbp-C0h]
  __int128 v65; // [rsp+10h] [rbp-C0h]
  __int64 v66; // [rsp+20h] [rbp-B0h]
  __int64 v67; // [rsp+20h] [rbp-B0h]
  __int64 v68; // [rsp+28h] [rbp-A8h]
  unsigned int v69; // [rsp+3Ch] [rbp-94h]
  __int128 v70; // [rsp+40h] [rbp-90h]
  unsigned __int64 v71; // [rsp+48h] [rbp-88h]
  unsigned __int64 v72; // [rsp+48h] [rbp-88h]
  __int64 v73; // [rsp+58h] [rbp-78h]
  unsigned int v74; // [rsp+60h] [rbp-70h] BYREF
  __int64 v75; // [rsp+68h] [rbp-68h]
  unsigned __int16 v76; // [rsp+70h] [rbp-60h] BYREF
  __int64 v77; // [rsp+78h] [rbp-58h]
  __int64 v78; // [rsp+80h] [rbp-50h] BYREF
  int v79; // [rsp+88h] [rbp-48h]
  __int64 v80; // [rsp+90h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v78, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    LOWORD(v8) = v79;
    LOWORD(v74) = v79;
    v75 = v80;
  }
  else
  {
    LODWORD(v8) = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v74 = v8;
    v75 = v44;
  }
  if ( (_WORD)v8 )
  {
    v9 = (unsigned __int16)v8;
    LOBYTE(v8) = (unsigned __int16)(v8 - 176) <= 0x34u;
    LODWORD(v10) = word_4456340[v9 - 1];
  }
  else
  {
    v10 = sub_3007240((__int64)&v74);
    v8 = HIDWORD(v10);
    HIDWORD(v73) = HIDWORD(v10);
  }
  v11 = *(const __m128i **)(a2 + 40);
  LODWORD(v73) = v10;
  BYTE4(v73) = v8;
  v12 = v11->m128i_u32[2];
  v13 = _mm_loadu_si128(v11);
  v66 = v11->m128i_i64[0];
  v14 = *(_QWORD *)(v11->m128i_i64[0] + 48) + 16 * v12;
  v64 = v11->m128i_u32[2];
  v15 = *(_WORD *)v14;
  v71 = v13.m128i_u64[1];
  v77 = *(_QWORD *)(v14 + 8);
  LODWORD(v14) = *(_DWORD *)(a2 + 24);
  v76 = v15;
  v69 = v14;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 17) > 0xD3u )
      goto LABEL_12;
  }
  else if ( !sub_30070B0((__int64)&v76) )
  {
    goto LABEL_12;
  }
  v16 = sub_3791F80(a1, a2);
  v20 = v19;
  v21 = v16;
  v22 = (__int64)v16;
  if ( v16 )
  {
    *(_QWORD *)&v23 = sub_379AB60(
                        (__int64)a1,
                        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
    v70 = v23;
    v24 = sub_379AB60((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
    v26 = *(_QWORD *)(a2 + 80);
    v27 = (_QWORD *)a1[1];
    v28 = v24;
    v29 = v25;
    v78 = v26;
    if ( v26 )
    {
      v68 = v25;
      v67 = v24;
      sub_B96E90((__int64)&v78, v26, 1);
      v28 = v67;
      v29 = v68;
    }
    *((_QWORD *)&v56 + 1) = v29;
    *(_QWORD *)&v56 = v28;
    v79 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v54 + 1) = v20;
    *(_QWORD *)&v54 = v21;
    v30 = sub_340F900(v27, v69, (__int64)&v78, v74, v75, v29, v54, v70, v56);
    goto LABEL_16;
  }
  if ( v76 )
  {
    LOWORD(v45) = word_4456580[v76 - 1];
  }
  else
  {
    v45 = sub_3009970((__int64)&v76, a2, 0, v17, v18);
    v61 = HIWORD(v45);
  }
  HIWORD(v46) = v61;
  LOWORD(v46) = v45;
  v60 = sub_327FD70(*(__int64 **)(a1[1] + 64), v46, v22, v73);
  v48 = v47;
  sub_2FE6CC0((__int64)&v78, *a1, *(_QWORD *)(a1[1] + 64), v76, v77);
  if ( (_BYTE)v78 == 7 )
  {
    v66 = sub_379AB60((__int64)a1, v13.m128i_u64[0], v13.m128i_i64[1]);
    v64 = v53;
    v71 = v53 | v13.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  sub_2FE6CC0((__int64)&v78, *a1, *(_QWORD *)(a1[1] + 64), v76, v77);
  if ( (_BYTE)v78 == 6 )
  {
    v51 = sub_3784B90((__int64)a1, a2);
    return sub_3790540((__int64)a1, (__int64)v51, v52, v74, v75, 0, v13);
  }
  v12 = v64;
  v49 = *(_QWORD *)(v66 + 48) + 16LL * v64;
  if ( *(_WORD *)v49 != (_WORD)v60 || *(_QWORD *)(v49 + 8) != v48 && !*(_WORD *)v49 )
  {
    v72 = v64 | v71 & 0xFFFFFFFF00000000LL;
    v66 = (__int64)sub_3790540((__int64)a1, v66, v72, v60, v48, 0, v13);
    v12 = v50;
    v71 = v50 | v72 & 0xFFFFFFFF00000000LL;
  }
  v11 = *(const __m128i **)(a2 + 40);
LABEL_12:
  *(_QWORD *)&v65 = sub_379AB60((__int64)a1, v11[2].m128i_u64[1], v11[3].m128i_i64[0]);
  *((_QWORD *)&v65 + 1) = v31;
  v32 = sub_379AB60((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v34 = (_QWORD *)a1[1];
  v35 = v32;
  v36 = v33;
  if ( v69 - 488 <= 1 )
  {
    v41 = *(_QWORD *)(a2 + 80);
    v42 = *(_QWORD *)(a2 + 40);
    v78 = v41;
    if ( v41 )
    {
      v63 = v33;
      v59 = v32;
      sub_B96E90((__int64)&v78, v41, 1);
      v35 = v59;
      v36 = v63;
    }
    v79 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v55 + 1) = v36;
    *(_QWORD *)&v55 = v35;
    v43 = sub_33FC130(
            v34,
            v69,
            (__int64)&v78,
            v74,
            v75,
            v36,
            __PAIR128__(v12 | v71 & 0xFFFFFFFF00000000LL, v66),
            v65,
            v55,
            *(_OWORD *)(v42 + 120));
    v38 = v78;
    v39 = (__int64)v43;
    if ( v78 )
      goto LABEL_17;
    return (unsigned __int8 *)v39;
  }
  v37 = *(_QWORD *)(a2 + 80);
  v78 = v37;
  if ( v37 )
  {
    v62 = v33;
    v58 = v32;
    sub_B96E90((__int64)&v78, v37, 1);
    v35 = v58;
    v36 = v62;
  }
  *((_QWORD *)&v57 + 1) = v36;
  *(_QWORD *)&v57 = v35;
  v79 = *(_DWORD *)(a2 + 72);
  v30 = sub_340F900(
          v34,
          v69,
          (__int64)&v78,
          v74,
          v75,
          v36,
          __PAIR128__(v12 | v71 & 0xFFFFFFFF00000000LL, v66),
          v65,
          v57);
LABEL_16:
  v38 = v78;
  v39 = v30;
  if ( v78 )
LABEL_17:
    sub_B91220((__int64)&v78, v38);
  return (unsigned __int8 *)v39;
}
