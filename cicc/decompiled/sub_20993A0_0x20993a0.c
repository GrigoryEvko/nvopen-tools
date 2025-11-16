// Function: sub_20993A0
// Address: 0x20993a0
//
void __fastcall sub_20993A0(
        unsigned __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  int v12; // r8d
  int v13; // r9d
  __int64 *v14; // r11
  __int64 v15; // r15
  int v16; // eax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 *v19; // rax
  __int64 v20; // rax
  unsigned int v21; // esi
  __int64 *v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // r12
  _QWORD *v25; // r14
  __int64 v26; // rax
  unsigned int v27; // edx
  unsigned __int8 v28; // al
  int v29; // r8d
  int v30; // r9d
  _QWORD *v31; // r14
  unsigned int v32; // edx
  __int64 v33; // r13
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  unsigned __int64 v37; // r10
  __int64 v38; // r8
  unsigned int k; // r9d
  __int64 v40; // rsi
  unsigned int i; // ecx
  __int64 v42; // rax
  unsigned int v43; // ecx
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // r13
  __int64 v47; // rax
  unsigned __int8 *v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // r15
  __int64 v51; // rdi
  __int64 v52; // rax
  unsigned int v53; // edx
  unsigned __int8 v54; // al
  _QWORD *v55; // rax
  _QWORD *v56; // r15
  __int64 v57; // rdx
  __int64 v58; // rsi
  __int64 v59; // rax
  int v60; // edx
  __int64 v61; // r11
  __int64 v62; // rsi
  __int64 v63; // rdx
  unsigned int v64; // esi
  __int64 v65; // r11
  unsigned int j; // ecx
  __int64 v67; // rax
  unsigned int v68; // ecx
  int v69; // r8d
  int v70; // esi
  __int64 v71; // r11
  __int64 v72; // rcx
  unsigned int v73; // edx
  unsigned int v74; // edx
  int v75; // edx
  int v76; // edx
  int v77; // edx
  int v78; // ecx
  int v79; // edi
  int v80; // edi
  int v81; // edx
  __int64 v82; // rcx
  unsigned int v83; // r9d
  int v84; // esi
  int v85; // edi
  __int64 v86; // [rsp+0h] [rbp-120h]
  __int64 v87; // [rsp+8h] [rbp-118h]
  unsigned __int64 v88; // [rsp+8h] [rbp-118h]
  unsigned __int64 v89; // [rsp+8h] [rbp-118h]
  __int64 *v90; // [rsp+10h] [rbp-110h]
  unsigned int v91; // [rsp+10h] [rbp-110h]
  int v92; // [rsp+10h] [rbp-110h]
  unsigned __int64 v93; // [rsp+10h] [rbp-110h]
  int v94; // [rsp+1Ch] [rbp-104h]
  __m128i v95; // [rsp+20h] [rbp-100h] BYREF
  __int64 v96; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v97; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v98; // [rsp+48h] [rbp-D8h]
  __int64 v99; // [rsp+50h] [rbp-D0h]
  unsigned __int64 v100; // [rsp+58h] [rbp-C8h]
  __m128i v101; // [rsp+60h] [rbp-C0h]
  unsigned __int64 v102; // [rsp+70h] [rbp-B0h]
  __int64 v103; // [rsp+78h] [rbp-A8h]
  __int64 v104; // [rsp+80h] [rbp-A0h]
  __int64 v105; // [rsp+88h] [rbp-98h]
  _QWORD *v106; // [rsp+90h] [rbp-90h]
  __int64 v107; // [rsp+98h] [rbp-88h]
  __int64 v108; // [rsp+A0h] [rbp-80h] BYREF
  int v109; // [rsp+A8h] [rbp-78h]
  __int128 v110; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v111; // [rsp+C0h] [rbp-60h]
  _QWORD v112[2]; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v113; // [rsp+E0h] [rbp-40h]
  unsigned __int64 v114; // [rsp+E8h] [rbp-38h]

  LODWORD(v96) = a3;
  v14 = sub_2051C20((__int64 *)a5, *(double *)a6.m128i_i64, a7, a8);
  v15 = (__int64)v14;
  v16 = *(unsigned __int16 *)(a1 + 24);
  v97 = v17;
  if ( v16 == 32 || v16 == 10 )
  {
    v20 = *(_QWORD *)(a1 + 88);
    v21 = *(_DWORD *)(v20 + 32);
    v22 = *(__int64 **)(v20 + 24);
    if ( v21 <= 0x40 )
      v23 = (__int64)((_QWORD)v22 << (64 - (unsigned __int8)v21)) >> (64 - (unsigned __int8)v21);
    else
      v23 = *v22;
    sub_2098400(a4, (__int64 *)a5, v23, a6, a7, a8);
    goto LABEL_12;
  }
  if ( v16 == 14 || v16 == 36 )
  {
    v25 = *(_QWORD **)(a5 + 552);
    v26 = sub_1E0A0C0(v25[4]);
    v27 = 8 * sub_15A9520(v26, *(_DWORD *)(v26 + 4));
    if ( v27 == 32 )
    {
      v28 = 5;
    }
    else if ( v27 > 0x20 )
    {
      v28 = 6;
      if ( v27 != 64 )
      {
        v28 = 0;
        if ( v27 == 128 )
          v28 = 7;
      }
    }
    else
    {
      v28 = 3;
      if ( v27 != 8 )
        v28 = 4 * (v27 == 16);
    }
    v31 = sub_1D299D0(v25, *(_DWORD *)(a1 + 84), v28, 0, 1);
    v33 = v32;
    v34 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v34 >= *(_DWORD *)(a4 + 12) )
    {
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v29, v30);
      v34 = *(unsigned int *)(a4 + 8);
    }
    v35 = (_QWORD *)(*(_QWORD *)a4 + 16 * v34);
    *v35 = v31;
    v35[1] = v33;
    ++*(_DWORD *)(a4 + 8);
    v24 = *(_QWORD *)(a5 + 552);
    if ( v15 )
      goto LABEL_13;
    goto LABEL_22;
  }
  if ( (_BYTE)v96 )
  {
    v18 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v18 >= *(_DWORD *)(a4 + 12) )
    {
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v12, v13);
      v18 = *(unsigned int *)(a4 + 8);
    }
    v19 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v18);
    *v19 = a1;
    v19[1] = a2;
    ++*(_DWORD *)(a4 + 8);
    goto LABEL_12;
  }
  v36 = *(unsigned int *)(a5 + 272);
  v94 = a2;
  v37 = v97;
  if ( (_DWORD)v36 )
  {
    LODWORD(v38) = v36 - 1;
    k = 1;
    v40 = *(_QWORD *)(a5 + 256);
    for ( i = (v36 - 1) & (a2 + ((a1 >> 9) ^ (a1 >> 4))); ; i = v38 & v43 )
    {
      v42 = v40 + 32LL * i;
      if ( a1 == *(_QWORD *)v42 )
      {
        if ( v94 == *(_DWORD *)(v42 + 8) )
        {
          if ( v42 != v40 + 32 * v36 )
          {
            v44 = *(unsigned int *)(v42 + 24);
            v96 = *(_QWORD *)(v42 + 16);
            v95.m128i_i64[0] = v96;
            v95.m128i_i64[1] = v44;
            if ( v96 )
              goto LABEL_35;
          }
          break;
        }
      }
      else if ( !*(_QWORD *)v42 && *(_DWORD *)(v42 + 8) == -1 )
      {
        break;
      }
      v43 = k + i;
      ++k;
    }
  }
  v48 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 16LL * (unsigned int)a2);
  v90 = v14;
  v86 = a5 + 248;
  v49 = sub_20989A0(a5 + 248, *v48, *((_QWORD *)v48 + 1), a5);
  v50 = *(_QWORD **)(a5 + 552);
  v51 = v50[4];
  LODWORD(v96) = *((_DWORD *)v49 + 21);
  v52 = sub_1E0A0C0(v51);
  v53 = 8 * sub_15A9520(v52, *(_DWORD *)(v52 + 4));
  if ( v53 == 32 )
  {
    v54 = 5;
  }
  else if ( v53 > 0x20 )
  {
    v54 = 6;
    if ( v53 != 64 )
    {
      v54 = 0;
      if ( v53 == 128 )
        v54 = 7;
    }
  }
  else
  {
    v54 = 3;
    if ( v53 != 8 )
      v54 = 4 * (v53 == 16);
  }
  v87 = (__int64)v90;
  v91 = v96;
  v55 = sub_1D299D0(v50, v96, v54, 0, 1);
  v56 = *(_QWORD **)(a5 + 552);
  v106 = v55;
  v96 = (__int64)v55;
  v95.m128i_i64[0] = (__int64)v55;
  v107 = v57;
  v112[0] = 0;
  v112[1] = 0;
  v113 = 0;
  v58 = v56[4];
  v95.m128i_i64[1] = (unsigned int)v57 | v95.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  sub_1E341E0((__int64)&v110, v58, v91, 0);
  v59 = *(_QWORD *)a5;
  v60 = *(_DWORD *)(a5 + 536);
  v108 = 0;
  v61 = v87;
  v109 = v60;
  if ( v59 )
  {
    if ( &v108 != (__int64 *)(v59 + 48) )
    {
      v62 = *(_QWORD *)(v59 + 48);
      v108 = v62;
      if ( v62 )
      {
        sub_1623A60((__int64)&v108, v62, 2);
        v61 = v87;
      }
    }
  }
  v104 = sub_1D2BF40(
           v56,
           v61,
           v97,
           (__int64)&v108,
           a1,
           a2,
           v95.m128i_i64[0],
           v95.m128i_i64[1],
           v110,
           v111,
           0,
           0,
           (__int64)v112);
  v105 = v63;
  v15 = v104;
  v37 = (unsigned int)v63 | v97 & 0xFFFFFFFF00000000LL;
  if ( v108 )
  {
    v88 = (unsigned int)v63 | v97 & 0xFFFFFFFF00000000LL;
    sub_161E7C0((__int64)&v108, v108);
    v37 = v88;
  }
  v64 = *(_DWORD *)(a5 + 272);
  if ( !v64 )
  {
    ++*(_QWORD *)(a5 + 248);
    goto LABEL_63;
  }
  v38 = *(_QWORD *)(a5 + 256);
  v65 = 0;
  v92 = 1;
  k = ((a1 >> 4) ^ (a1 >> 9)) + a2;
  for ( j = (v64 - 1) & k; ; j = (v64 - 1) & v68 )
  {
    v67 = v38 + 32LL * j;
    if ( a1 != *(_QWORD *)v67 )
      break;
    if ( v94 == *(_DWORD *)(v67 + 8) )
      goto LABEL_60;
LABEL_54:
    v68 = v92 + j;
    ++v92;
  }
  if ( *(_QWORD *)v67 )
    goto LABEL_54;
  v75 = *(_DWORD *)(v67 + 8);
  if ( v75 != -1 )
  {
    if ( !v65 && v75 == -2 )
      v65 = v38 + 32LL * j;
    goto LABEL_54;
  }
  v78 = *(_DWORD *)(a5 + 264);
  if ( v65 )
    v67 = v65;
  ++*(_QWORD *)(a5 + 248);
  v76 = v78 + 1;
  if ( 4 * (v78 + 1) < 3 * v64 )
  {
    if ( v64 - *(_DWORD *)(a5 + 268) - v76 > v64 >> 3 )
      goto LABEL_75;
    v89 = v37;
    sub_2099180(v86, v64);
    v79 = *(_DWORD *)(a5 + 272);
    if ( v79 )
    {
      v80 = v79 - 1;
      v67 = 0;
      v81 = 1;
      v38 = *(_QWORD *)(a5 + 256);
      v37 = v89;
      for ( k = v80 & (((a1 >> 4) ^ (a1 >> 9)) + a2); ; k = v80 & v83 )
      {
        v82 = v38 + 32LL * k;
        if ( a1 == *(_QWORD *)v82 )
        {
          if ( v94 == *(_DWORD *)(v82 + 8) )
          {
            v76 = *(_DWORD *)(a5 + 264) + 1;
            v67 = v38 + 32LL * k;
            goto LABEL_75;
          }
        }
        else if ( !*(_QWORD *)v82 )
        {
          v84 = *(_DWORD *)(v82 + 8);
          if ( v84 == -1 )
          {
            if ( !v67 )
              v67 = v38 + 32LL * k;
            v76 = *(_DWORD *)(a5 + 264) + 1;
            goto LABEL_75;
          }
          if ( !v67 && v84 == -2 )
            v67 = v38 + 32LL * k;
        }
        v83 = v81 + k;
        ++v81;
      }
    }
LABEL_107:
    ++*(_DWORD *)(a5 + 264);
    BUG();
  }
LABEL_63:
  v93 = v37;
  sub_2099180(v86, 2 * v64);
  v69 = *(_DWORD *)(a5 + 272);
  if ( !v69 )
    goto LABEL_107;
  LODWORD(v38) = v69 - 1;
  v70 = 1;
  v71 = *(_QWORD *)(a5 + 256);
  v72 = 0;
  v37 = v93;
  v73 = v38 & (a2 + ((a1 >> 9) ^ (a1 >> 4)));
  while ( 2 )
  {
    v67 = v71 + 32LL * v73;
    if ( a1 == *(_QWORD *)v67 )
    {
      if ( v94 == *(_DWORD *)(v67 + 8) )
      {
        v76 = *(_DWORD *)(a5 + 264) + 1;
        goto LABEL_75;
      }
      goto LABEL_67;
    }
    if ( *(_QWORD *)v67 )
    {
LABEL_67:
      v74 = v70 + v73;
      ++v70;
      v73 = v38 & v74;
      continue;
    }
    break;
  }
  v85 = *(_DWORD *)(v67 + 8);
  if ( v85 != -1 )
  {
    if ( !v72 && v85 == -2 )
      v72 = v71 + 32LL * v73;
    goto LABEL_67;
  }
  if ( v72 )
    v67 = v72;
  v76 = *(_DWORD *)(a5 + 264) + 1;
LABEL_75:
  *(_DWORD *)(a5 + 264) = v76;
  if ( *(_QWORD *)v67 || *(_DWORD *)(v67 + 8) != -1 )
    --*(_DWORD *)(a5 + 268);
  v103 = a2;
  v102 = a1;
  *(_QWORD *)v67 = a1;
  v77 = v103;
  *(_QWORD *)(v67 + 16) = 0;
  *(_DWORD *)(v67 + 8) = v77;
  *(_DWORD *)(v67 + 24) = 0;
LABEL_60:
  v101 = _mm_load_si128(&v95);
  *(_QWORD *)(v67 + 16) = v95.m128i_i64[0];
  *(_DWORD *)(v67 + 24) = v101.m128i_i32[2];
LABEL_35:
  v113 = v15;
  v114 = v37;
  v95.m128i_i64[0] = v96;
  v45 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v45 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v38, k);
    v45 = *(unsigned int *)(a4 + 8);
  }
  v46 = v97;
  *(__m128i *)(*(_QWORD *)a4 + 16 * v45) = _mm_load_si128(&v95);
  v47 = (unsigned int)v114;
  ++*(_DWORD *)(a4 + 8);
  v113 = v15;
  v97 = v47 | v46 & 0xFFFFFFFF00000000LL;
LABEL_12:
  v24 = *(_QWORD *)(a5 + 552);
  if ( v15 )
  {
LABEL_13:
    nullsub_686();
    v99 = v15;
    v100 = v97;
    *(_QWORD *)(v24 + 176) = v15;
    *(_DWORD *)(v24 + 184) = v100;
    sub_1D23870();
    return;
  }
LABEL_22:
  v98 = v97;
  *(_QWORD *)(v24 + 176) = 0;
  *(_DWORD *)(v24 + 184) = v98;
}
