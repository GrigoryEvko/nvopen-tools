// Function: sub_3304AC0
// Address: 0x3304ac0
//
__int64 __fastcall sub_3304AC0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  unsigned int v4; // r15d
  __int64 v5; // rax
  bool v6; // zf
  unsigned __int16 v7; // dx
  __int64 v8; // rax
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v12; // rbx
  int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 v18; // r9
  __int64 v19; // rsi
  __m128i v20; // xmm4
  __int64 v21; // rcx
  unsigned __int16 v22; // dx
  unsigned __int16 v23; // r8
  __int64 v24; // rdi
  int v25; // eax
  __int64 v26; // r13
  __int64 v27; // r10
  __m128i v28; // rax
  unsigned int v29; // eax
  int v30; // r9d
  unsigned __int64 v31; // r15
  int v32; // ebx
  __int64 v33; // rsi
  __int64 v34; // r14
  __int64 *v35; // rdx
  __int64 v36; // r14
  __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdx
  int v45; // r9d
  __m128i v46; // rax
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned __int16 *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int128 v52; // [rsp-10h] [rbp-2B0h]
  __int128 v53; // [rsp+0h] [rbp-2A0h]
  __int64 v54; // [rsp+8h] [rbp-298h]
  __int64 *v55; // [rsp+18h] [rbp-288h]
  unsigned __int64 v56; // [rsp+28h] [rbp-278h]
  __int64 v57; // [rsp+28h] [rbp-278h]
  __int64 v58; // [rsp+30h] [rbp-270h]
  unsigned int v59; // [rsp+70h] [rbp-230h]
  unsigned int v60; // [rsp+74h] [rbp-22Ch]
  __int64 v61; // [rsp+78h] [rbp-228h]
  unsigned int v62; // [rsp+80h] [rbp-220h]
  unsigned int v63; // [rsp+84h] [rbp-21Ch]
  unsigned __int16 v64; // [rsp+88h] [rbp-218h]
  int v65; // [rsp+88h] [rbp-218h]
  __int64 v66; // [rsp+90h] [rbp-210h]
  unsigned int v67; // [rsp+98h] [rbp-208h]
  unsigned int v68; // [rsp+98h] [rbp-208h]
  __int64 v69; // [rsp+98h] [rbp-208h]
  __int64 v70; // [rsp+A0h] [rbp-200h]
  __int64 v71; // [rsp+A0h] [rbp-200h]
  unsigned int v72; // [rsp+A8h] [rbp-1F8h]
  __int64 v73; // [rsp+B0h] [rbp-1F0h]
  int v74; // [rsp+B0h] [rbp-1F0h]
  __int64 v75; // [rsp+B0h] [rbp-1F0h]
  __int64 v76; // [rsp+B8h] [rbp-1E8h]
  unsigned int v77; // [rsp+C0h] [rbp-1E0h]
  __int64 v78; // [rsp+C0h] [rbp-1E0h]
  __int16 v79; // [rsp+C8h] [rbp-1D8h]
  __m128i v80; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v81; // [rsp+E0h] [rbp-1C0h]
  __int64 v82; // [rsp+E8h] [rbp-1B8h]
  __int64 v83; // [rsp+F8h] [rbp-1A8h]
  __m128i v84; // [rsp+100h] [rbp-1A0h] BYREF
  __m128i v85; // [rsp+110h] [rbp-190h] BYREF
  __m128i v86; // [rsp+120h] [rbp-180h] BYREF
  __int64 v87; // [rsp+130h] [rbp-170h] BYREF
  int v88; // [rsp+138h] [rbp-168h]
  __int64 v89; // [rsp+140h] [rbp-160h] BYREF
  int v90; // [rsp+148h] [rbp-158h]
  __int64 v91; // [rsp+150h] [rbp-150h]
  __int64 v92; // [rsp+158h] [rbp-148h]
  __int128 v93; // [rsp+160h] [rbp-140h] BYREF
  __int64 v94; // [rsp+170h] [rbp-130h]
  _OWORD v95[2]; // [rsp+180h] [rbp-120h] BYREF
  unsigned __int64 v96[2]; // [rsp+1A0h] [rbp-100h] BYREF
  _BYTE v97[32]; // [rsp+1B0h] [rbp-F0h] BYREF
  _BYTE *v98; // [rsp+1D0h] [rbp-D0h] BYREF
  __int64 v99; // [rsp+1D8h] [rbp-C8h]
  _BYTE v100[64]; // [rsp+1E0h] [rbp-C0h] BYREF
  __m128i v101; // [rsp+220h] [rbp-80h] BYREF
  _BYTE v102[112]; // [rsp+230h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_QWORD *)v2;
  v4 = *(_DWORD *)(v2 + 8);
  v5 = *(_QWORD *)(a2 + 48);
  v6 = *(_DWORD *)(v3 + 24) == 298;
  v7 = *(_WORD *)v5;
  v8 = *(_QWORD *)(v5 + 8);
  v84.m128i_i16[0] = v7;
  v84.m128i_i64[1] = v8;
  if ( !v6 )
    return 0;
  if ( (*(_BYTE *)(v3 + 33) & 0xC) != 0 )
    return 0;
  if ( (*(_WORD *)(v3 + 32) & 0x380) != 0 )
    return 0;
  v10 = *(_QWORD *)(v3 + 56);
  if ( !v10 )
    return 0;
  v12 = a2;
  v13 = 1;
  do
  {
    if ( v4 == *(_DWORD *)(v10 + 8) )
    {
      if ( !v13 )
        return 0;
      v10 = *(_QWORD *)(v10 + 32);
      if ( !v10 )
        goto LABEL_15;
      if ( v4 == *(_DWORD *)(v10 + 8) )
        return 0;
      v13 = 0;
    }
    v10 = *(_QWORD *)(v10 + 32);
  }
  while ( v10 );
  if ( v13 == 1 )
    return 0;
LABEL_15:
  if ( (*(_BYTE *)(*(_QWORD *)(v3 + 112) + 37LL) & 0xF) != 0 )
    return 0;
  v73 = 16LL * v4;
  v14 = *(_QWORD *)(v3 + 48) + v73;
  v79 = *(_WORD *)v14;
  v80.m128i_i64[0] = *(_QWORD *)(v14 + 8);
  if ( (*(_BYTE *)(v3 + 32) & 8) != 0 )
    return 0;
  if ( !v7 )
  {
    if ( sub_30070B0((__int64)&v84) )
    {
      v15 = sub_3007240((__int64)&v84);
      v83 = v15;
      goto LABEL_20;
    }
    return 0;
  }
  if ( (unsigned __int16)(v7 - 17) > 0xD3u )
    return 0;
  LODWORD(v15) = word_4456340[v7 - 1];
LABEL_20:
  v77 = v15 & (v15 - 1);
  if ( v77 )
    return 0;
  v16 = a1[1];
  v17 = *(__int64 (**)())(*(_QWORD *)v16 + 1584LL);
  if ( v17 == sub_2FE3520 || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v17)(v16, a2, 0) )
    return 0;
  v54 = a1[1];
  v18 = *(unsigned int *)(a2 + 24);
  v96[0] = (unsigned __int64)v97;
  v19 = v84.m128i_i64[1];
  v96[1] = 0x400000000LL;
  if ( !(unsigned __int8)sub_32611B0(v84.m128i_u32[0], v84.m128i_i64[1], v12, v3, v4, v18, (__int64)v96, v54) )
    goto LABEL_52;
  v6 = *(_DWORD *)(v12 + 24) == 213;
  v20 = _mm_loadu_si128(&v84);
  v85.m128i_i16[0] = v79;
  v21 = v80.m128i_i64[0];
  v63 = !v6 + 2;
  v80.m128i_i64[0] = v3;
  v85.m128i_i64[1] = v21;
  v86 = v20;
  while ( 1 )
  {
    v22 = v86.m128i_i16[0];
    if ( v86.m128i_i16[0] )
    {
      v23 = v85.m128i_i16[0];
      if ( v85.m128i_i16[0] )
      {
        v24 = a1[1];
        v25 = ((int)*(unsigned __int16 *)(v24 + 2 * (v85.m128i_u16[0] + 274LL * v86.m128i_u16[0] + 71704) + 6) >> (4 * v63))
            & 0xF;
        if ( !v25 || (_BYTE)v25 == 4 )
        {
          v26 = v80.m128i_i64[0];
          goto LABEL_32;
        }
      }
    }
    if ( (unsigned int)sub_3281500(&v85, v19) <= 1 )
      break;
    sub_33D0340(&v101, *a1, &v86);
    v19 = *a1;
    v86 = _mm_loadu_si128(&v101);
    sub_33D0340(&v101, v19, &v85);
    v85 = _mm_loadu_si128(&v101);
  }
  v26 = v80.m128i_i64[0];
  v24 = a1[1];
  v22 = v86.m128i_i16[0];
  v23 = v85.m128i_i16[0];
LABEL_32:
  if ( sub_325F4A0(v24, v63, v22, v23) )
  {
    v87 = *(_QWORD *)(v12 + 80);
    if ( v87 )
    {
      v80.m128i_i64[0] = v27;
      sub_325F5D0(&v87);
    }
    v88 = *(_DWORD *)(v12 + 72);
    v72 = sub_3281500(&v84, v63);
    v67 = sub_3281500(&v86, v63);
    v60 = v72 / v67;
    v28.m128i_i64[0] = sub_3285A00((unsigned __int16 *)&v85);
    v101 = v28;
    v29 = sub_CA1930(&v101);
    v98 = v100;
    v101.m128i_i64[0] = (__int64)v102;
    v101.m128i_i64[1] = 0x400000000LL;
    v99 = 0x400000000LL;
    v62 = v29;
    v80 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v26 + 40) + 40LL));
    if ( v72 >= v67 )
    {
      v59 = v4;
      v31 = v56;
      v61 = v29;
      v68 = 0;
      v57 = v3;
      v58 = v12;
      v32 = v64;
      do
      {
        v33 = *(_QWORD *)(v26 + 112);
        v34 = *a1;
        v95[0] = _mm_loadu_si128((const __m128i *)(v33 + 40));
        v95[1] = _mm_loadu_si128((const __m128i *)(v33 + 56));
        LOBYTE(v32) = *(_BYTE *)(v33 + 34);
        v65 = *(unsigned __int16 *)(v33 + 32);
        BYTE1(v32) = 1;
        sub_327C6E0((__int64)&v93, (__int64 *)v33, v68);
        v35 = *(__int64 **)(v26 + 40);
        v89 = *(_QWORD *)(v26 + 80);
        if ( v89 )
        {
          v55 = v35;
          sub_325F5D0(&v89);
          v35 = v55;
        }
        v90 = *(_DWORD *)(v26 + 72);
        v36 = sub_33F1DB0(
                v34,
                v63,
                (unsigned int)&v89,
                v86.m128i_i32[0],
                v86.m128i_i32[2],
                v32,
                *v35,
                v35[1],
                v80.m128i_i64[0],
                v80.m128i_i64[1],
                v93,
                v94,
                v85.m128i_i64[0],
                v85.m128i_i64[1],
                v65,
                (__int64)v95);
        if ( v89 )
          sub_B91220((__int64)&v89, v89);
        v37 = *a1;
        LOBYTE(v92) = 0;
        v91 = v61;
        v81 = sub_3409320(v37, v80.m128i_i32[0], v80.m128i_i32[2], v61, v92, (unsigned int)&v87, 0);
        v80.m128i_i64[0] = v81;
        v82 = v38;
        v70 &= 0xFFFFFFFF00000000LL;
        v80.m128i_i64[1] = (unsigned int)v38 | v80.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        sub_3050D50((__int64)&v98, v36, v70, 0xFFFFFFFF00000000LL, v39, v40);
        v31 = v31 & 0xFFFFFFFF00000000LL | 1;
        sub_3050D50((__int64)&v101, v36, v31, v41, v42, v43);
        ++v77;
        v68 += v62;
      }
      while ( v60 > v77 );
      v4 = v59;
      v12 = v58;
      v3 = v57;
    }
    *((_QWORD *)&v53 + 1) = v101.m128i_u32[2];
    *(_QWORD *)&v53 = v101.m128i_i64[0];
    v78 = sub_33FC220(*a1, 2, (unsigned int)&v87, 1, 0, v30, v53);
    v71 = v44;
    *((_QWORD *)&v52 + 1) = (unsigned int)v99;
    *(_QWORD *)&v52 = v98;
    v46.m128i_i64[0] = sub_33FC220(*a1, 159, (unsigned int)&v87, v84.m128i_i32[0], v84.m128i_i32[2], v45, v52);
    v80 = v46;
    sub_32B3E80((__int64)a1, v78, 1, 0, v47, v48);
    v95[0] = _mm_load_si128(&v80);
    sub_32EB790((__int64)a1, v12, (__int64 *)v95, 1, 1);
    v49 = (unsigned __int16 *)(*(_QWORD *)(v26 + 48) + v73);
    v66 = *a1;
    v69 = *((_QWORD *)v49 + 1);
    v74 = *v49;
    sub_3285E70((__int64)v95, v3);
    v75 = sub_33FAF80(v66, 216, (unsigned int)v95, v74, v69, v66, *(_OWORD *)&v80);
    v76 = v50;
    sub_9C6650(v95);
    sub_3304760(a1, (__int64)v96, v26, v4, v80.m128i_i64[0], v80.m128i_i64[1], *(_DWORD *)(v12 + 24));
    sub_32EFDE0((__int64)a1, v26, v75, v76, v78, v71, 1);
    v51 = v12;
    if ( (_BYTE *)v101.m128i_i64[0] != v102 )
    {
      v80.m128i_i64[0] = v12;
      _libc_free(v101.m128i_u64[0]);
      v51 = v80.m128i_i64[0];
    }
    if ( v98 != v100 )
    {
      v80.m128i_i64[0] = v51;
      _libc_free((unsigned __int64)v98);
      v51 = v80.m128i_i64[0];
    }
    v80.m128i_i64[0] = v51;
    sub_9C6650(&v87);
    result = v80.m128i_i64[0];
  }
  else
  {
LABEL_52:
    result = 0;
  }
  if ( (_BYTE *)v96[0] != v97 )
  {
    v80.m128i_i64[0] = result;
    _libc_free(v96[0]);
    return v80.m128i_i64[0];
  }
  return result;
}
