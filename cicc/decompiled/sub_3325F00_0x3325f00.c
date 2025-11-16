// Function: sub_3325F00
// Address: 0x3325f00
//
__int64 __fastcall sub_3325F00(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __int64 v8; // r14
  __int64 v9; // r10
  __int64 v10; // r15
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  int v22; // eax
  int v23; // edx
  int v24; // r15d
  int v25; // r14d
  unsigned int v26; // edx
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r10
  __int64 v31; // rdx
  int v32; // r14d
  int v33; // edx
  int v34; // r15d
  __int16 v35; // ax
  unsigned __int16 *v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 v40; // rsi
  __int128 v41; // rax
  __int64 v42; // r8
  __int64 v43; // r10
  __int64 v44; // rsi
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 v47; // rdi
  int v48; // edx
  __int64 v49; // rdx
  __int64 v50; // [rsp-10h] [rbp-190h]
  __int64 v51; // [rsp-8h] [rbp-188h]
  __int64 v52; // [rsp+8h] [rbp-178h]
  __int64 v53; // [rsp+10h] [rbp-170h]
  int v54; // [rsp+10h] [rbp-170h]
  __int64 v55; // [rsp+18h] [rbp-168h]
  __int64 v56; // [rsp+18h] [rbp-168h]
  int v57; // [rsp+20h] [rbp-160h]
  unsigned int v58; // [rsp+28h] [rbp-158h]
  __int64 v59; // [rsp+28h] [rbp-158h]
  char v60; // [rsp+28h] [rbp-158h]
  __int64 v61; // [rsp+40h] [rbp-140h]
  char v62; // [rsp+40h] [rbp-140h]
  unsigned int v63; // [rsp+48h] [rbp-138h]
  unsigned int v64; // [rsp+48h] [rbp-138h]
  __int64 v65; // [rsp+50h] [rbp-130h]
  __int64 v66; // [rsp+50h] [rbp-130h]
  __int64 v67; // [rsp+58h] [rbp-128h]
  __int128 v68; // [rsp+60h] [rbp-120h]
  int v69; // [rsp+60h] [rbp-120h]
  __int128 v70; // [rsp+60h] [rbp-120h]
  __int64 v71; // [rsp+A0h] [rbp-E0h] BYREF
  int v72; // [rsp+A8h] [rbp-D8h]
  unsigned int v73; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v74; // [rsp+B8h] [rbp-C8h]
  __int64 v75; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v76; // [rsp+C8h] [rbp-B8h]
  __int128 v77; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+E0h] [rbp-A0h]
  __int128 v79; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v80; // [rsp+100h] [rbp-80h]
  __int128 v81; // [rsp+110h] [rbp-70h] BYREF
  __int64 v82; // [rsp+120h] [rbp-60h]
  __int64 v83; // [rsp+130h] [rbp-50h] BYREF
  __int64 v84; // [rsp+138h] [rbp-48h]
  __int64 v85; // [rsp+140h] [rbp-40h]
  __int64 v86; // [rsp+148h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v7 = _mm_loadu_si128((const __m128i *)v4 + 5);
  v8 = *v4;
  v9 = *v4;
  v71 = v5;
  v10 = v4[1];
  v11 = *((unsigned int *)v4 + 2);
  v12 = v4[5];
  v13 = *((unsigned int *)v4 + 12);
  if ( v5 )
  {
    v58 = *((_DWORD *)v4 + 12);
    v61 = v4[5];
    v63 = *((_DWORD *)v4 + 2);
    v65 = v9;
    sub_B96E90((__int64)&v71, v5, 1);
    v13 = v58;
    v12 = v61;
    v11 = v63;
    v9 = v65;
  }
  v14 = *(_QWORD *)(v9 + 48) + 16 * v11;
  v15 = *(_QWORD *)(v12 + 48) + 16 * v13;
  v16 = *(_QWORD *)(a1 + 16);
  v72 = *(_DWORD *)(a2 + 72);
  v17 = *(_QWORD *)(v14 + 8);
  LOWORD(v73) = *(_WORD *)v14;
  v74 = v17;
  v18 = *(_QWORD *)(v15 + 8);
  LOWORD(v75) = *(_WORD *)v15;
  v76 = v18;
  v19 = sub_33EDFE0(v16, v73, v17, 1);
  v67 = v20;
  v66 = v19;
  v64 = *(_DWORD *)(v19 + 96);
  sub_2EAC300((__int64)&v77, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 40LL), v64, 0);
  v21 = *(_QWORD *)(a1 + 16);
  v62 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v21 + 40) + 48LL) + 8LL)
                 + 40LL * (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v21 + 40) + 48LL) + 32LL) + v64)
                 + 16);
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v22 = sub_33F4560(v21, (int)v21 + 288, 0, (unsigned int)&v71, v8, v10, v66, v67, v77, v78, v62, 0, (__int64)&v83);
  v24 = v23;
  v25 = v22;
  *(_QWORD *)&v68 = sub_33FB960(*(_QWORD *)(a1 + 16), v7.m128i_i64[0], v7.m128i_i64[1]);
  *((_QWORD *)&v68 + 1) = v26 | v7.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v59 = sub_3007410(
          (__int64)&v75,
          *(__int64 **)(*(_QWORD *)(a1 + 16) + 64LL),
          v26,
          0xFFFFFFFF00000000LL,
          (__int64)&v75,
          v27);
  v28 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 16) + 40LL));
  v60 = sub_AE5260(v28, v59);
  if ( !(_WORD)v75 )
  {
    if ( sub_30070B0((__int64)&v75) )
      goto LABEL_5;
LABEL_10:
    v40 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)&v41 = sub_3466750(*(_QWORD *)(a1 + 8), v40, v66, v67, v73, v74, v68, *((__int64 *)&v68 + 1));
    v83 = 0;
    v43 = *(_QWORD *)(a1 + 16);
    v70 = v41;
    v84 = 0;
    v85 = 0;
    v86 = 0;
    if ( (_WORD)v73 )
    {
      v44 = v43;
      v45 = 0;
      LOWORD(v46) = word_4456580[(unsigned __int16)v73 - 1];
    }
    else
    {
      v54 = v43;
      v46 = sub_3009970((__int64)&v73, v40, v50, v51, v42);
      v44 = *(_QWORD *)(a1 + 16);
      LODWORD(v43) = v54;
      v52 = v46;
      v45 = v49;
    }
    v47 = v52;
    v57 = v43;
    v56 = v45;
    LOWORD(v47) = v46;
    sub_2EAC3A0((__int64)&v79, *(__int64 **)(v44 + 40));
    v32 = sub_33F5040(
            v57,
            v25,
            v24,
            (unsigned int)&v71,
            v6.m128i_i32[0],
            v6.m128i_i32[2],
            v70,
            *((__int64 *)&v70 + 1),
            v79,
            v80,
            v47,
            v56,
            v60,
            0,
            (__int64)&v83);
    v34 = v48;
    goto LABEL_6;
  }
  if ( (unsigned __int16)(v75 - 17) > 0xD3u )
    goto LABEL_10;
LABEL_5:
  v29 = sub_3465D80(*(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 16), v66, v67, v73, v74, v75, v76, v68);
  v30 = *(_QWORD *)(a1 + 16);
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v53 = v29;
  v55 = v31;
  v69 = v30;
  sub_2EAC3A0((__int64)&v81, *(__int64 **)(v30 + 40));
  v32 = sub_33F4560(
          v69,
          v25,
          v24,
          (unsigned int)&v71,
          v6.m128i_i32[0],
          v6.m128i_i32[2],
          v53,
          v55,
          v81,
          v82,
          v60,
          0,
          (__int64)&v83);
  v34 = v33;
LABEL_6:
  HIBYTE(v35) = 1;
  LOBYTE(v35) = v62;
  v36 = *(unsigned __int16 **)(a2 + 48);
  v83 = 0;
  v37 = *(_QWORD *)(a1 + 16);
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v38 = sub_33F1F00(
          v37,
          *v36,
          *((_QWORD *)v36 + 1),
          (unsigned int)&v71,
          v32,
          v34,
          v66,
          v67,
          v77,
          v78,
          v35,
          0,
          (__int64)&v83,
          0);
  if ( v71 )
    sub_B91220((__int64)&v71, v71);
  return v38;
}
