// Function: sub_115A4C0
// Address: 0x115a4c0
//
unsigned __int8 *__fastcall sub_115A4C0(const __m128i *a1, unsigned __int8 *a2)
{
  int v3; // eax
  __int64 v4; // rbx
  int v5; // ecx
  __int64 v6; // r14
  unsigned __int8 *v7; // r15
  __int64 v8; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r15
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __m128i v30; // xmm5
  unsigned __int64 v31; // xmm6_8
  __int64 v32; // rsi
  __m128i v33; // xmm7
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  char v37; // al
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rax
  __m128i v42; // xmm0
  __m128i v43; // xmm1
  __int64 v44; // rbx
  unsigned __int64 v45; // xmm2_8
  __m128i v46; // xmm3
  __int64 v47; // rax
  int v48; // eax
  int v49; // r14d
  __int64 v50; // rax
  __m128i v51; // xmm5
  unsigned __int64 v52; // xmm6_8
  __int64 v53; // rsi
  __m128i v54; // xmm7
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rsi
  int v59; // [rsp+4h] [rbp-FCh]
  int v60; // [rsp+8h] [rbp-F8h]
  __int64 v61; // [rsp+8h] [rbp-F8h]
  int v62; // [rsp+8h] [rbp-F8h]
  int v63; // [rsp+8h] [rbp-F8h]
  int v64; // [rsp+8h] [rbp-F8h]
  int v65; // [rsp+8h] [rbp-F8h]
  __int64 v66; // [rsp+18h] [rbp-E8h] BYREF
  unsigned __int64 v67; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v68; // [rsp+28h] [rbp-D8h]
  unsigned int v69; // [rsp+30h] [rbp-D0h]
  __int64 v70; // [rsp+38h] [rbp-C8h]
  unsigned int v71; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v72; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v73; // [rsp+58h] [rbp-A8h]
  __int64 v74; // [rsp+60h] [rbp-A0h]
  __int64 v75; // [rsp+68h] [rbp-98h]
  __int64 *v76; // [rsp+70h] [rbp-90h]
  __m128i v77; // [rsp+80h] [rbp-80h] BYREF
  __m128i v78; // [rsp+90h] [rbp-70h]
  unsigned __int64 *v79; // [rsp+A0h] [rbp-60h]
  unsigned __int8 *v80; // [rsp+A8h] [rbp-58h]
  __m128i v81; // [rsp+B0h] [rbp-50h]
  __int64 v82; // [rsp+C0h] [rbp-40h]

  v3 = *a2;
  v4 = *((_QWORD *)a2 - 8);
  v5 = v3 - 29;
  if ( (_BYTE)v3 != 47 )
    goto LABEL_2;
  v10 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 16LL);
  if ( !v10 || *(_QWORD *)(v10 + 8) )
  {
    v6 = *((_QWORD *)a2 - 4);
  }
  else
  {
    v59 = v5;
    v61 = *((_QWORD *)a2 - 8);
    v37 = sub_920620(v61);
    v6 = *((_QWORD *)a2 - 4);
    v5 = v59;
    if ( v37 )
    {
      if ( (*(_BYTE *)(v61 + 1) & 2) != 0 && *(_BYTE *)v61 == 85 )
      {
        v38 = *(_QWORD *)(v61 - 32);
        if ( v38 )
        {
          if ( !*(_BYTE *)v38 && *(_QWORD *)(v38 + 24) == *(_QWORD *)(v61 + 80) && *(_DWORD *)(v38 + 36) == 285 )
          {
            v39 = *(_DWORD *)(v61 + 4) & 0x7FFFFFF;
            v16 = *(_QWORD *)(v61 - 32 * v39);
            if ( v16 )
            {
              v40 = *(_QWORD *)(v61 + 32 * (1 - v39));
              if ( v40 )
              {
                v66 = *(_QWORD *)(v61 + 32 * (1 - v39));
                if ( v6 == v16 )
                  goto LABEL_68;
              }
            }
          }
        }
      }
    }
  }
  v11 = *(_QWORD *)(v6 + 16);
  if ( !v11 || *(_QWORD *)(v11 + 8) )
  {
    v12 = *((_QWORD *)a2 - 8);
    goto LABEL_14;
  }
  v60 = v5;
  v22 = sub_920620(v6);
  v12 = *((_QWORD *)a2 - 8);
  v5 = v60;
  if ( !v22
    || (v4 = *((_QWORD *)a2 - 8), (*(_BYTE *)(v6 + 1) & 2) == 0)
    || *(_BYTE *)v6 != 85
    || (v57 = *(_QWORD *)(v6 - 32)) == 0
    || *(_BYTE *)v57
    || *(_QWORD *)(v57 + 24) != *(_QWORD *)(v6 + 80)
    || *(_DWORD *)(v57 + 36) != 285
    || (v58 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF, (v16 = *(_QWORD *)(v6 - 32 * v58)) == 0)
    || (v40 = *(_QWORD *)(v6 + 32 * (1 - v58))) == 0 )
  {
    v6 = *((_QWORD *)a2 - 4);
LABEL_14:
    v4 = v12;
    if ( v5 != 18 )
      goto LABEL_4;
    goto LABEL_15;
  }
  v66 = *(_QWORD *)(v6 + 32 * (1 - v58));
  if ( v12 != v16 )
  {
LABEL_2:
    v6 = *((_QWORD *)a2 - 4);
    goto LABEL_3;
  }
LABEL_68:
  v62 = v5;
  v41 = sub_AD64C0(*(_QWORD *)(v40 + 8), 1, 0);
  v42 = _mm_loadu_si128(a1 + 6);
  v43 = _mm_loadu_si128(a1 + 7);
  v44 = v41;
  v45 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v46 = _mm_loadu_si128(a1 + 9);
  v72 = v41 & 0xFFFFFFFFFFFFFFFBLL;
  v79 = (unsigned __int64 *)v45;
  v80 = a2;
  v67 = v66 & 0xFFFFFFFFFFFFFFFBLL;
  v47 = a1[10].m128i_i64[0];
  LODWORD(v74) = 1;
  v73 = 0;
  LODWORD(v76) = 1;
  v75 = 0;
  v69 = 1;
  v68 = 0;
  v71 = 1;
  v70 = 0;
  v82 = v47;
  v77 = v42;
  v78 = v43;
  v81 = v46;
  v48 = sub_9B0100((__int64 *)&v67, (__int64 *)&v72, &v77);
  v5 = v62;
  v49 = v48;
  if ( v71 > 0x40 && v70 )
  {
    j_j___libc_free_0_0(v70);
    v5 = v62;
  }
  if ( v69 > 0x40 && v68 )
  {
    v63 = v5;
    j_j___libc_free_0_0(v68);
    v5 = v63;
  }
  if ( (unsigned int)v76 > 0x40 && v75 )
  {
    v64 = v5;
    j_j___libc_free_0_0(v75);
    v5 = v64;
  }
  if ( (unsigned int)v74 > 0x40 && v73 )
  {
    v65 = v5;
    j_j___libc_free_0_0(v73);
    v5 = v65;
  }
  if ( v49 == 3 )
  {
    v19 = a1[2].m128i_i64[0];
    v17 = v66;
    v18 = v44;
LABEL_29:
    v20 = v16;
    goto LABEL_30;
  }
  v6 = *((_QWORD *)a2 - 4);
  v4 = *((_QWORD *)a2 - 8);
LABEL_3:
  if ( v5 == 18 )
  {
LABEL_15:
    if ( !sub_B44680((__int64)a2) )
      return 0;
    v13 = sub_920620(v4);
    if ( !v4 )
      return 0;
    if ( !v13 )
      return 0;
    if ( (*(_BYTE *)(v4 + 1) & 2) == 0 )
      return 0;
    if ( *(_BYTE *)v4 != 85 )
      return 0;
    v14 = *(_QWORD *)(v4 - 32);
    if ( !v14 )
      return 0;
    if ( *(_BYTE *)v14 )
      return 0;
    if ( *(_QWORD *)(v14 + 24) != *(_QWORD *)(v4 + 80) )
      return 0;
    if ( *(_DWORD *)(v14 + 36) != 285 )
      return 0;
    v15 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
    v16 = *(_QWORD *)(v4 - 32 * v15);
    if ( !v16 )
      return 0;
    if ( !*(_QWORD *)(v4 + 32 * (1 - v15)) )
      return 0;
    v66 = *(_QWORD *)(v4 + 32 * (1 - v15));
    v77.m128i_i32[0] = 285;
    v77.m128i_i32[2] = 0;
    v78.m128i_i64[0] = v16;
    v78.m128i_i32[2] = 1;
    v79 = &v72;
    if ( !sub_1159100((__int64)&v77, v6) )
      return 0;
    v17 = v66;
    v18 = v72;
    if ( *(_QWORD *)(v72 + 8) != *(_QWORD *)(v66 + 8) )
      return 0;
    v19 = a1[2].m128i_i64[0];
    goto LABEL_29;
  }
LABEL_4:
  v7 = 0;
  if ( v5 == 21 && sub_B451B0((__int64)a2) && sub_B451C0((__int64)a2) )
  {
    v74 = v6;
    LODWORD(v72) = 285;
    LODWORD(v73) = 0;
    LODWORD(v75) = 1;
    v76 = &v66;
    v8 = *(_QWORD *)(v4 + 16);
    if ( !v8 )
      return 0;
    if ( *(_QWORD *)(v8 + 8) )
    {
LABEL_36:
      if ( !v8 )
        return 0;
      if ( *(_QWORD *)(v8 + 8) )
        return 0;
      if ( !(unsigned __int8)sub_920620(v4) )
        return 0;
      if ( (*(_BYTE *)(v4 + 1) & 2) == 0 )
        return 0;
      if ( *(_BYTE *)v4 != 85 )
        return 0;
      v23 = *(_QWORD *)(v4 - 32);
      if ( !v23 )
        return 0;
      if ( *(_BYTE *)v23 )
        return 0;
      if ( *(_QWORD *)(v23 + 24) != *(_QWORD *)(v4 + 80) )
        return 0;
      if ( *(_DWORD *)(v23 + 36) != 285 )
        return 0;
      v24 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
      v25 = *(_QWORD *)(v4 - 32 * v24);
      if ( !v25 )
        return 0;
      if ( !*(_QWORD *)(v4 + 32 * (1 - v24)) )
        return 0;
      v66 = *(_QWORD *)(v4 + 32 * (1 - v24));
      v26 = sub_920620(v6);
      if ( !v6 || !v26 || (*(_BYTE *)(v6 + 1) & 2) == 0 || *(_BYTE *)v6 != 47 )
        return 0;
      v27 = *(_QWORD *)(v6 - 64);
      v28 = *(_QWORD *)(v6 - 32);
      if ( v25 == v27 )
      {
        if ( !v28 )
          return 0;
        v72 = *(_QWORD *)(v6 - 32);
      }
      else
      {
        if ( !v27 || v28 != v25 )
          return 0;
        v72 = *(_QWORD *)(v6 - 64);
      }
      v29 = sub_AD64C0(*(_QWORD *)(v66 + 8), 1, 0);
      v30 = _mm_loadu_si128(a1 + 7);
      v31 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v32 = v29;
      v33 = _mm_loadu_si128(a1 + 9);
      v34 = a1[10].m128i_i64[0];
      v77 = _mm_loadu_si128(a1 + 6);
      v79 = (unsigned __int64 *)v31;
      v82 = v34;
      v80 = a2;
      v78 = v30;
      v81 = v33;
      if ( (unsigned int)sub_9AFB10(v66, v32, &v77) == 3 )
      {
        v35 = sub_AD62B0(*(_QWORD *)(v66 + 8));
        v36 = sub_11554C0((__int64)a2, a1[2].m128i_i64[0], v25, v66, v35);
        LOWORD(v79) = 257;
        v7 = (unsigned __int8 *)sub_B504D0(21, v36, v72, (__int64)&v77, 0, 0);
        sub_B45260(v7, (__int64)a2, 1);
        return v7;
      }
      return 0;
    }
    if ( !sub_1159100((__int64)&v72, v4) )
      goto LABEL_83;
    v50 = sub_AD64C0(*(_QWORD *)(v66 + 8), 1, 0);
    v51 = _mm_loadu_si128(a1 + 7);
    v52 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
    v53 = v50;
    v54 = _mm_loadu_si128(a1 + 9);
    v55 = a1[10].m128i_i64[0];
    v77 = _mm_loadu_si128(a1 + 6);
    v79 = (unsigned __int64 *)v52;
    v82 = v55;
    v80 = a2;
    v78 = v51;
    v81 = v54;
    if ( (unsigned int)sub_9AFB10(v66, v53, &v77) != 3 )
    {
LABEL_83:
      v8 = *(_QWORD *)(v4 + 16);
      goto LABEL_36;
    }
    v56 = sub_AD62B0(*(_QWORD *)(v66 + 8));
    v19 = a1[2].m128i_i64[0];
    v17 = v66;
    v20 = v6;
    v18 = v56;
LABEL_30:
    v21 = sub_11554C0((__int64)a2, v19, v20, v17, v18);
    return sub_F162A0((__int64)a1, (__int64)a2, v21);
  }
  return v7;
}
