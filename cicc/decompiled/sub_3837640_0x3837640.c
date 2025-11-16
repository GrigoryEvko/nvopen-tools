// Function: sub_3837640
// Address: 0x3837640
//
unsigned __int8 *__fastcall sub_3837640(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // r10
  __int64 v4; // rax
  unsigned __int64 v5; // r8
  __m128i v6; // xmm0
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int16 v10; // r12
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned __int8 *v14; // rax
  __int64 v15; // r10
  __int64 v16; // rdx
  unsigned __int8 *v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int16 v23; // r12
  __int64 v24; // r13
  __int64 v25; // rax
  unsigned int v26; // edx
  unsigned __int8 *v27; // rax
  __int64 v28; // r9
  __int64 v29; // r10
  unsigned __int8 *v30; // r12
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // rsi
  _QWORD *v34; // rbx
  unsigned __int16 *v35; // rdx
  __int64 v36; // r15
  __int64 v37; // rcx
  __int64 v38; // rsi
  unsigned __int8 *v39; // rax
  __int64 v40; // rsi
  unsigned __int8 *v41; // r12
  unsigned __int8 *v43; // r12
  __int64 v44; // rdx
  __int64 v45; // r13
  __int128 v46; // rax
  __int64 v47; // r10
  _QWORD *v48; // r15
  __int128 v49; // kr00_16
  __int64 v50; // rsi
  __int64 v51; // rbx
  __int64 v52; // rcx
  unsigned int v53; // esi
  unsigned __int8 *v54; // rax
  __int128 v55; // [rsp-30h] [rbp-C0h]
  __int128 v56; // [rsp-20h] [rbp-B0h]
  __int64 v57; // [rsp+0h] [rbp-90h]
  unsigned __int64 v58; // [rsp+8h] [rbp-88h]
  __int64 v59; // [rsp+8h] [rbp-88h]
  __int64 v60; // [rsp+10h] [rbp-80h]
  __int64 v61; // [rsp+10h] [rbp-80h]
  __int64 v62; // [rsp+10h] [rbp-80h]
  unsigned __int64 v63; // [rsp+18h] [rbp-78h]
  __int64 v64; // [rsp+18h] [rbp-78h]
  __int64 v65; // [rsp+18h] [rbp-78h]
  __int64 v66; // [rsp+20h] [rbp-70h]
  __int64 v67; // [rsp+20h] [rbp-70h]
  __int128 v68; // [rsp+20h] [rbp-70h]
  __int64 v69; // [rsp+20h] [rbp-70h]
  __int128 v70; // [rsp+30h] [rbp-60h]
  __int64 v71; // [rsp+30h] [rbp-60h]
  __int128 v73; // [rsp+40h] [rbp-50h]
  __int64 v74; // [rsp+50h] [rbp-40h] BYREF
  int v75; // [rsp+58h] [rbp-38h]

  v3 = a2;
  v4 = *(_QWORD *)(a2 + 40);
  if ( *(_DWORD *)(a2 + 64) == 2 )
  {
    v43 = sub_37AF270(a1, *(_QWORD *)v4, *(_QWORD *)(v4 + 8), a3);
    v45 = v44;
    *(_QWORD *)&v46 = sub_37AF270(
                        a1,
                        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                        a3);
    v47 = a2;
    v48 = *(_QWORD **)(a1 + 8);
    v49 = v46;
    v50 = *(_QWORD *)(a2 + 80);
    v51 = *(_QWORD *)(*((_QWORD *)v43 + 6) + 16LL * (unsigned int)v45 + 8);
    v52 = *(unsigned __int16 *)(*((_QWORD *)v43 + 6) + 16LL * (unsigned int)v45);
    v74 = v50;
    if ( v50 )
    {
      v69 = v52;
      v71 = a2;
      v73 = v46;
      sub_B96E90((__int64)&v74, v50, 1);
      v52 = v69;
      v47 = v71;
      v49 = v73;
    }
    v53 = *(_DWORD *)(v47 + 24);
    *((_QWORD *)&v56 + 1) = v45;
    *(_QWORD *)&v56 = v43;
    v75 = *(_DWORD *)(v47 + 72);
    v54 = sub_3406EB0(v48, v53, (__int64)&v74, v52, v51, *((__int64 *)&v49 + 1), v56, v49);
    v40 = v74;
    v41 = v54;
    if ( v74 )
      goto LABEL_13;
  }
  else
  {
    v5 = *(_QWORD *)v4;
    v6 = _mm_loadu_si128((const __m128i *)(v4 + 80));
    v7 = *(_QWORD *)(v4 + 8);
    v8 = *(_QWORD *)(*(_QWORD *)v4 + 80LL);
    v70 = (__int128)_mm_loadu_si128((const __m128i *)(v4 + 120));
    v9 = *(_QWORD *)(*(_QWORD *)v4 + 48LL) + 16LL * *(unsigned int *)(v4 + 8);
    v10 = *(_WORD *)v9;
    v11 = *(_QWORD *)(v9 + 8);
    v74 = v8;
    if ( v8 )
    {
      v63 = v5;
      v66 = v3;
      sub_B96E90((__int64)&v74, v8, 1);
      v5 = v63;
      v3 = v66;
    }
    v67 = v3;
    v75 = *(_DWORD *)(v5 + 72);
    v12 = sub_37AE0F0(a1, v5, v7);
    v14 = sub_3400810(
            *(_QWORD **)(a1 + 8),
            v12,
            v7 & 0xFFFFFFFF00000000LL | v13,
            v6.m128i_i64[0],
            v6.m128i_i64[1],
            (__int64)&v74,
            v6,
            v70,
            v10,
            v11);
    v15 = v67;
    v64 = v16;
    v17 = v14;
    if ( v74 )
    {
      sub_B91220((__int64)&v74, v74);
      v15 = v67;
    }
    *(_QWORD *)&v68 = v17;
    *((_QWORD *)&v68 + 1) = v64;
    v18 = *(_QWORD *)(v15 + 40);
    v19 = *(_QWORD *)(v18 + 40);
    v20 = *(_QWORD *)(v18 + 48);
    v21 = *(_QWORD *)(v19 + 80);
    v22 = *(_QWORD *)(v19 + 48) + 16LL * *(unsigned int *)(v18 + 48);
    v23 = *(_WORD *)v22;
    v24 = *(_QWORD *)(v22 + 8);
    v74 = v21;
    if ( v21 )
    {
      v60 = v15;
      v57 = v20;
      v58 = v19;
      sub_B96E90((__int64)&v74, v21, 1);
      v20 = v57;
      v19 = v58;
      v15 = v60;
    }
    v59 = v15;
    v61 = v20;
    v75 = *(_DWORD *)(v19 + 72);
    v25 = sub_37AE0F0(a1, v19, v20);
    v27 = sub_3400810(
            *(_QWORD **)(a1 + 8),
            v25,
            v61 & 0xFFFFFFFF00000000LL | v26,
            v6.m128i_i64[0],
            v6.m128i_i64[1],
            (__int64)&v74,
            v6,
            v70,
            v23,
            v24);
    v29 = v59;
    v30 = v27;
    v32 = v31;
    if ( v74 )
    {
      sub_B91220((__int64)&v74, v74);
      v29 = v59;
    }
    v33 = *(_QWORD *)(v29 + 80);
    v34 = *(_QWORD **)(a1 + 8);
    v35 = (unsigned __int16 *)(*((_QWORD *)v17 + 6) + 16LL * (unsigned int)v64);
    v36 = *((_QWORD *)v35 + 1);
    v37 = *v35;
    v74 = v33;
    if ( v33 )
    {
      v62 = v37;
      v65 = v29;
      sub_B96E90((__int64)&v74, v33, 1);
      v37 = v62;
      v29 = v65;
    }
    v38 = *(unsigned int *)(v29 + 24);
    *((_QWORD *)&v55 + 1) = v32;
    *(_QWORD *)&v55 = v30;
    v75 = *(_DWORD *)(v29 + 72);
    v39 = sub_33FC130(v34, v38, (__int64)&v74, v37, v36, v28, v68, v55, *(_OWORD *)&v6, v70);
    v40 = v74;
    v41 = v39;
    if ( v74 )
LABEL_13:
      sub_B91220((__int64)&v74, v40);
  }
  return v41;
}
