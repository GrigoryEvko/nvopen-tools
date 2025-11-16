// Function: sub_1FA77F0
// Address: 0x1fa77f0
//
__int64 *__fastcall sub_1FA77F0(__int64 a1, _QWORD *a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r15
  unsigned int v10; // r14d
  __int64 v11; // rax
  unsigned __int8 v12; // bl
  const void **v13; // rdx
  __int64 v14; // rax
  int v15; // eax
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  int v24; // ecx
  __int64 v25; // rsi
  unsigned __int8 v26; // al
  __int64 v27; // r9
  __int64 v28; // rsi
  __int64 v29; // rcx
  __int64 v30; // r11
  __int64 v31; // r8
  __int64 *v32; // rbx
  __m128i v33; // rax
  __m128i v34; // xmm1
  __int64 v35; // rsi
  __int64 v36; // r11
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 *v39; // r11
  __int64 v40; // r9
  __int64 v41; // r10
  __int64 v42; // rcx
  const void **v43; // r8
  __int32 v44; // edx
  __int64 v45; // rsi
  __int128 *v46; // rbx
  __int64 *v47; // r13
  __int64 v48; // r15
  __int64 v49; // rsi
  _QWORD *v50; // rax
  __m128i v51; // xmm0
  __int64 v52; // rax
  char v53; // r14
  __int64 v54; // rax
  unsigned int v55; // ebx
  unsigned int v56; // eax
  __int64 v57; // rsi
  __int64 *v58; // r13
  __int64 *v59; // rax
  __int64 v60; // rsi
  __int64 *v61; // r13
  __int128 v62; // [rsp-10h] [rbp-F0h]
  __int128 v63; // [rsp-10h] [rbp-F0h]
  __int64 v64; // [rsp+8h] [rbp-D8h]
  __int64 v65; // [rsp+10h] [rbp-D0h]
  __int64 v66; // [rsp+18h] [rbp-C8h]
  __int64 v67; // [rsp+20h] [rbp-C0h]
  __int64 v68; // [rsp+20h] [rbp-C0h]
  __int64 v69; // [rsp+28h] [rbp-B8h]
  __int64 v70; // [rsp+30h] [rbp-B0h]
  __int64 *v71; // [rsp+30h] [rbp-B0h]
  const void **v72; // [rsp+30h] [rbp-B0h]
  __int64 v73; // [rsp+38h] [rbp-A8h]
  __m128i v74; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v75; // [rsp+50h] [rbp-90h]
  unsigned int v76; // [rsp+60h] [rbp-80h] BYREF
  const void **v77; // [rsp+68h] [rbp-78h]
  __int64 v78; // [rsp+70h] [rbp-70h] BYREF
  int v79; // [rsp+78h] [rbp-68h]
  __int64 v80; // [rsp+80h] [rbp-60h] BYREF
  int v81; // [rsp+88h] [rbp-58h]
  __m128i v82; // [rsp+90h] [rbp-50h] BYREF
  __int64 v83; // [rsp+A0h] [rbp-40h]
  int v84; // [rsp+A8h] [rbp-38h]

  v5 = (__int64)a2;
  v6 = (_QWORD *)a2[4];
  v7 = *v6;
  v8 = v6[1];
  v9 = *v6;
  v10 = *((_DWORD *)v6 + 2);
  v11 = a2[5];
  v12 = *(_BYTE *)v11;
  v13 = *(const void ***)(v11 + 8);
  v14 = a2[6];
  LOBYTE(v76) = v12;
  v77 = v13;
  if ( v14 && !*(_QWORD *)(v14 + 32) && *(_WORD *)(*(_QWORD *)(v14 + 16) + 24LL) == 154 )
    return 0;
  v15 = *(unsigned __int16 *)(v9 + 24);
  v75.m128i_i64[0] = (__int64)v13;
  if ( v15 == 11
    || v15 == 33
    || (v74.m128i_i64[0] = v7,
        v74.m128i_i64[1] = v8,
        v16 = sub_1D16930(v9),
        v8 = v74.m128i_i64[1],
        v7 = v74.m128i_i64[0],
        v17 = v75.m128i_i64[0],
        v16) )
  {
    v18 = a2[9];
    v19 = *(__int64 **)a1;
    v82.m128i_i64[0] = v18;
    if ( v18 )
    {
      v75.m128i_i64[0] = v7;
      v75.m128i_i64[1] = v8;
      sub_1623A60((__int64)&v82, v18, 2);
      v8 = v75.m128i_i64[1];
      v7 = v75.m128i_i64[0];
    }
    *((_QWORD *)&v62 + 1) = v8;
    *(_QWORD *)&v62 = v7;
    v82.m128i_i32[2] = *(_DWORD *)(v5 + 64);
    v20 = sub_1D309E0(v19, 157, (__int64)&v82, v76, v77, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v62);
    goto LABEL_9;
  }
  v24 = *(unsigned __int16 *)(v9 + 24);
  if ( (_WORD)v24 != 160 )
  {
    if ( v24 != 154 )
    {
      if ( (_WORD)v24 == 185
        && (*(_BYTE *)(v9 + 27) & 0xC) == 0
        && (*(_WORD *)(v9 + 26) & 0x380) == 0
        && sub_1D18C00(v9, 1, v10) )
      {
        v25 = 16LL * v10 + *(_QWORD *)(v9 + 40);
        v75.m128i_i64[0] = 16LL * v10;
        v26 = *(_BYTE *)v25;
        if ( *(_BYTE *)v25 )
        {
          if ( v12 && !((unsigned __int8)*(_WORD *)(*(_QWORD *)(a1 + 8) + 2 * (v26 + 115LL * v12 + 16104)) >> 4) )
          {
            v27 = *(_QWORD *)(v25 + 8);
            v28 = *(_QWORD *)(v5 + 72);
            v29 = *(_QWORD *)(v9 + 104);
            v30 = *(_QWORD *)(v9 + 32);
            v31 = v26;
            v82.m128i_i64[0] = v28;
            v32 = *(__int64 **)a1;
            if ( v28 )
            {
              v67 = v26;
              v69 = v27;
              v73 = v29;
              v74.m128i_i64[0] = v30;
              sub_1623A60((__int64)&v82, v28, 2);
              v31 = v67;
              v27 = v69;
              v29 = v73;
              v30 = v74.m128i_i64[0];
            }
            v82.m128i_i32[2] = *(_DWORD *)(v5 + 64);
            v33.m128i_i64[0] = sub_1D2B590(
                                 v32,
                                 1,
                                 (__int64)&v82,
                                 v76,
                                 (__int64)v77,
                                 v29,
                                 *(_OWORD *)v30,
                                 *(_QWORD *)(v30 + 40),
                                 *(_QWORD *)(v30 + 48),
                                 v31,
                                 v27);
            v74 = v33;
            if ( v82.m128i_i64[0] )
              sub_161E7C0((__int64)&v82, v82.m128i_i64[0]);
            v34 = _mm_load_si128(&v74);
            v82 = v34;
            sub_1F994A0(a1, v5, v82.m128i_i64, 1, 1);
            v35 = *(_QWORD *)(v9 + 72);
            v36 = *(_QWORD *)a1;
            v68 = v74.m128i_i64[0];
            v80 = v35;
            if ( v35 )
            {
              v70 = v36;
              sub_1623A60((__int64)&v80, v35, 2);
              v36 = v70;
            }
            v71 = (__int64 *)v36;
            v81 = *(_DWORD *)(v9 + 64);
            v37 = sub_1D38E70(v36, 1, (__int64)&v80, 0, a3, *(double *)v34.m128i_i64, a5);
            v39 = v71;
            v40 = v37;
            v41 = v38;
            v42 = *(unsigned __int8 *)(*(_QWORD *)(v9 + 40) + v75.m128i_i64[0]);
            v43 = *(const void ***)(*(_QWORD *)(v9 + 40) + v75.m128i_i64[0] + 8);
            v78 = *(_QWORD *)(v9 + 72);
            if ( v78 )
            {
              v66 = v38;
              v64 = v42;
              v72 = v43;
              v75.m128i_i64[0] = (__int64)v39;
              v65 = v37;
              sub_1623A60((__int64)&v78, v78, 2);
              v42 = v64;
              v40 = v65;
              v41 = v66;
              v43 = v72;
              v39 = (__int64 *)v75.m128i_i64[0];
            }
            *((_QWORD *)&v63 + 1) = v41;
            *(_QWORD *)&v63 = v40;
            v79 = *(_DWORD *)(v9 + 64);
            v82.m128i_i64[0] = (__int64)sub_1D332F0(
                                          v39,
                                          154,
                                          (__int64)&v78,
                                          v42,
                                          v43,
                                          0,
                                          *(double *)a3.m128i_i64,
                                          *(double *)v34.m128i_i64,
                                          a5,
                                          v74.m128i_i64[0],
                                          v74.m128i_u64[1],
                                          v63);
            v82.m128i_i32[2] = v44;
            v83 = v68;
            v84 = 1;
            sub_1F994A0(a1, v9, v82.m128i_i64, 2, 1);
            if ( v78 )
              sub_161E7C0((__int64)&v78, v78);
            if ( v80 )
              sub_161E7C0((__int64)&v80, v80);
            return (__int64 *)v5;
          }
        }
      }
      return sub_1F77270((__int64 **)a1, v5, *(double *)a3.m128i_i64, a4, a5);
    }
    v48 = *(_QWORD *)(v9 + 32);
    v49 = *(_QWORD *)(*(_QWORD *)(v48 + 40) + 88LL);
    v50 = *(_QWORD **)(v49 + 24);
    if ( *(_DWORD *)(v49 + 32) > 0x40u )
      v50 = (_QWORD *)*v50;
    if ( v50 != (_QWORD *)1 )
      return sub_1F77270((__int64 **)a1, v5, *(double *)a3.m128i_i64, a4, a5);
    v51 = _mm_loadu_si128((const __m128i *)v48);
    v52 = *(_QWORD *)(*(_QWORD *)v48 + 40LL) + 16LL * *(unsigned int *)(v48 + 8);
    v53 = *(_BYTE *)v52;
    v75 = v51;
    v54 = *(_QWORD *)(v52 + 8);
    if ( v53 == v12 )
    {
      if ( v54 == v17 || v12 )
        return (__int64 *)v75.m128i_i64[0];
      v82.m128i_i8[0] = 0;
      v82.m128i_i64[1] = v54;
    }
    else
    {
      v82.m128i_i8[0] = v53;
      v82.m128i_i64[1] = v54;
      if ( v12 )
      {
        v55 = sub_1F6C8D0(v12);
LABEL_49:
        if ( v53 )
          v56 = sub_1F6C8D0(v53);
        else
          v56 = sub_1F58D40((__int64)&v82);
        if ( v56 > v55 )
        {
          v57 = *(_QWORD *)(v5 + 72);
          v58 = *(__int64 **)a1;
          v82.m128i_i64[0] = v57;
          if ( v57 )
            sub_1623A60((__int64)&v82, v57, 2);
          v82.m128i_i32[2] = *(_DWORD *)(v5 + 64);
          v59 = sub_1D332F0(
                  v58,
                  154,
                  (__int64)&v82,
                  v76,
                  v77,
                  0,
                  *(double *)v51.m128i_i64,
                  a4,
                  a5,
                  v75.m128i_i64[0],
                  v75.m128i_u64[1],
                  *(_OWORD *)(v48 + 40));
          v22 = v82.m128i_i64[0];
          v5 = (__int64)v59;
          if ( v82.m128i_i64[0] )
            goto LABEL_10;
          return (__int64 *)v5;
        }
        v60 = *(_QWORD *)(v5 + 72);
        v61 = *(__int64 **)a1;
        v82.m128i_i64[0] = v60;
        if ( v60 )
          sub_1623A60((__int64)&v82, v60, 2);
        v82.m128i_i32[2] = *(_DWORD *)(v5 + 64);
        v20 = sub_1D309E0(
                v61,
                157,
                (__int64)&v82,
                v76,
                v77,
                0,
                *(double *)v51.m128i_i64,
                a4,
                *(double *)a5.m128i_i64,
                *(_OWORD *)&v75);
        goto LABEL_9;
      }
    }
    v55 = sub_1F58D40((__int64)&v76);
    goto LABEL_49;
  }
  if ( !v12 || *(_BYTE *)(*(_QWORD *)(a1 + 8) + 259LL * v12 + 2582) )
    return sub_1F77270((__int64 **)a1, v5, *(double *)a3.m128i_i64, a4, a5);
  v45 = a2[9];
  v46 = *(__int128 **)(v9 + 32);
  v47 = *(__int64 **)a1;
  v82.m128i_i64[0] = v45;
  if ( v45 )
    sub_1623A60((__int64)&v82, v45, 2);
  v82.m128i_i32[2] = *(_DWORD *)(v5 + 64);
  v20 = sub_1D309E0(v47, 160, (__int64)&v82, v76, v77, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, *v46);
LABEL_9:
  v22 = v82.m128i_i64[0];
  v5 = v20;
  if ( v82.m128i_i64[0] )
  {
LABEL_10:
    v75.m128i_i64[0] = v21;
    sub_161E7C0((__int64)&v82, v22);
  }
  return (__int64 *)v5;
}
