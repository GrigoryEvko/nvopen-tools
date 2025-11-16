// Function: sub_21D7E10
// Address: 0x21d7e10
//
__int64 *__fastcall sub_21D7E10(__int64 a1, __int64 a2, __int64 *a3, __m128 a4, double a5, __m128i a6)
{
  int v8; // r11d
  __int16 v9; // ax
  unsigned __int8 *v10; // rdx
  unsigned int v11; // r13d
  __int64 *result; // rax
  __int64 v13; // rsi
  __int64 *v14; // r14
  unsigned __int8 *v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // r15
  unsigned int v18; // ebx
  __int64 *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rax
  unsigned __int64 v23; // rdx
  __int128 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rbx
  unsigned __int64 v27; // r11
  unsigned int v28; // r14d
  int v29; // esi
  __int64 v30; // rcx
  __int16 v31; // ax
  unsigned int v32; // esi
  __int64 v33; // rdi
  _DWORD *v34; // rdx
  int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // rbx
  unsigned int v38; // r14d
  __int64 v39; // r13
  const __m128i *v40; // rdx
  __int64 v41; // r8
  __int64 v42; // rcx
  __int64 v43; // rsi
  __int64 v44; // r11
  __int32 v45; // r9d
  int v46; // eax
  __int64 *v47; // rbx
  __m128i v48; // xmm0
  const void ***v49; // rax
  int v50; // edx
  __int64 v51; // r9
  __int64 v52; // rcx
  _QWORD *v53; // rdx
  __int64 v54; // rdx
  _QWORD *v55; // rax
  char v56; // r8
  __int64 v57; // rax
  __int64 v58; // rsi
  unsigned __int8 *v59; // rax
  const void **v60; // r8
  __int64 v61; // rcx
  unsigned int v62; // edx
  _DWORD *v63; // rax
  __int64 v64; // r10
  __int128 v65; // [rsp-20h] [rbp-D0h]
  __int128 v66; // [rsp-10h] [rbp-C0h]
  __int128 v67; // [rsp-10h] [rbp-C0h]
  __int32 v68; // [rsp+4h] [rbp-ACh]
  __int64 v69; // [rsp+8h] [rbp-A8h]
  __int64 v70; // [rsp+8h] [rbp-A8h]
  int v71; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v72; // [rsp+18h] [rbp-98h]
  __int64 v73; // [rsp+20h] [rbp-90h]
  __int64 v74; // [rsp+20h] [rbp-90h]
  const void **v75; // [rsp+20h] [rbp-90h]
  unsigned __int64 v76; // [rsp+28h] [rbp-88h]
  const void **v77; // [rsp+30h] [rbp-80h]
  const void **v78; // [rsp+30h] [rbp-80h]
  int v79; // [rsp+30h] [rbp-80h]
  __int64 *v80; // [rsp+30h] [rbp-80h]
  __int64 *v81; // [rsp+30h] [rbp-80h]
  __int64 v82; // [rsp+40h] [rbp-70h] BYREF
  int v83; // [rsp+48h] [rbp-68h]
  __int64 v84; // [rsp+50h] [rbp-60h] BYREF
  int v85; // [rsp+58h] [rbp-58h]
  __int64 v86; // [rsp+60h] [rbp-50h]
  __int32 v87; // [rsp+68h] [rbp-48h]
  __m128i v88; // [rsp+70h] [rbp-40h]

  v8 = sub_1700720(*(_QWORD *)(a1 + 8));
  v9 = *(_WORD *)(a2 + 24);
  if ( v9 == 76 )
  {
LABEL_35:
    v36 = *(__int64 **)(a2 + 32);
    v79 = v8;
    v37 = *v36;
    v73 = v36[1];
    v38 = *((_DWORD *)v36 + 12);
    v39 = v36[5];
    result = sub_21D7B80(a2, *v36, v73, v39, v36[6], a3[2], a4, a5, a6, v8);
    if ( !result )
      return sub_21D7B80(a2, v39, v38, v37, v73, a3[2], a4, a5, a6, v79);
    return result;
  }
  if ( v9 > 76 )
  {
    if ( v9 != 122 )
    {
      if ( v9 == 137 )
      {
        v10 = *(unsigned __int8 **)(a2 + 40);
        v11 = *v10;
        v77 = (const void **)*((_QWORD *)v10 + 1);
        if ( *v10 != 15 )
          return 0;
        v40 = *(const __m128i **)(a2 + 32);
        v41 = v40->m128i_i64[0];
        v42 = v40->m128i_u32[2];
        if ( *(_BYTE *)(*(_QWORD *)(v40->m128i_i64[0] + 40) + 16 * v42) != 86 )
          return 0;
        v43 = *(_QWORD *)(a2 + 72);
        v44 = v40[2].m128i_i64[1];
        v45 = v40[3].m128i_i32[0];
        v82 = v43;
        if ( v43 )
        {
          v68 = v45;
          v69 = v44;
          v71 = v42;
          v74 = v41;
          sub_1623A60((__int64)&v82, v43, 2);
          v40 = *(const __m128i **)(a2 + 32);
          v45 = v68;
          v44 = v69;
          LODWORD(v42) = v71;
          v41 = v74;
        }
        v85 = v42;
        v46 = *(_DWORD *)(a2 + 64);
        v47 = (__int64 *)a3[2];
        v87 = v45;
        v84 = v41;
        v86 = v44;
        v48 = _mm_loadu_si128(v40 + 5);
        v88 = v48;
        v83 = v46;
        v49 = (const void ***)sub_1D252B0((__int64)v47, 2, 0, 2, 0);
        *((_QWORD *)&v66 + 1) = 3;
        *(_QWORD *)&v66 = &v84;
        *((_QWORD *)&v65 + 1) = 1;
        *(_QWORD *)&v65 = sub_1D36D80(v47, 298, (__int64)&v82, v49, v50, *(double *)v48.m128i_i64, a5, a6, v51, v66);
        result = sub_1D332F0(
                   (__int64 *)a3[2],
                   104,
                   (__int64)&v82,
                   v11,
                   v77,
                   0,
                   *(double *)v48.m128i_i64,
                   a5,
                   a6,
                   v65,
                   0,
                   v65);
        if ( v82 )
        {
          v80 = result;
          sub_161E7C0((__int64)&v82, v82);
          return v80;
        }
        return result;
      }
      if ( v9 == 118 )
      {
        v25 = *(__int64 **)(a2 + 32);
        v26 = *v25;
        v27 = v25[1];
        v28 = *((_DWORD *)v25 + 2);
        v29 = *(unsigned __int16 *)(*v25 + 24);
        v30 = v25[5];
        v31 = *(_WORD *)(*v25 + 24);
        if ( (_WORD)v29 == 32 || v29 == 10 )
        {
          v28 = *((_DWORD *)v25 + 12);
          v31 = *(_WORD *)(v30 + 24);
          v26 = v25[5];
          v30 = *v25;
        }
        v32 = 0;
        v33 = 0;
        if ( v31 == 144 )
        {
          v34 = *(_DWORD **)(v26 + 32);
          v33 = v26;
          v32 = v28;
          v28 = v34[2];
          v26 = *(_QWORD *)v34;
          v27 = v28 | v27 & 0xFFFFFFFF00000000LL;
          v31 = *(_WORD *)(*(_QWORD *)v34 + 24LL);
        }
        if ( v31 == -615 )
        {
          v63 = *(_DWORD **)(v26 + 32);
          v64 = *(_QWORD *)v63;
          v28 = v63[2];
          v31 = *(_WORD *)(*(_QWORD *)v63 + 24LL);
          v26 = v64;
          v27 = v28 | v27 & 0xFFFFFFFF00000000LL;
        }
        if ( (unsigned int)(unsigned __int16)v31 - 659 > 1 )
          return 0;
        v35 = *(unsigned __int16 *)(v30 + 24);
        if ( v35 != 10 && v35 != 32 )
          return 0;
        v52 = *(_QWORD *)(v30 + 88);
        v53 = *(_QWORD **)(v52 + 24);
        if ( *(_DWORD *)(v52 + 32) > 0x40u )
          v53 = (_QWORD *)*v53;
        if ( v53 != (_QWORD *)255 )
          return 0;
        if ( (unsigned __int16)(v31 - 44) <= 1u )
        {
          if ( (*(_BYTE *)(v26 + 26) & 2) != 0 )
          {
LABEL_59:
            if ( (unsigned __int8)(*(_BYTE *)(v26 + 88) - 25) <= 1u )
            {
              v54 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v26 + 32) + 40LL * (unsigned int)(*(_DWORD *)(v26 + 56) - 1))
                              + 88LL);
              v55 = *(_QWORD **)(v54 + 24);
              if ( *(_DWORD *)(v54 + 32) > 0x40u )
                v55 = (_QWORD *)*v55;
              if ( (_DWORD)v55 != 2 )
              {
                v56 = 0;
                if ( v33 )
                {
                  v81 = (__int64 *)a3[2];
                  v57 = v32;
                  v58 = *(_QWORD *)(a2 + 72);
                  v59 = (unsigned __int8 *)(*(_QWORD *)(v33 + 40) + 16 * v57);
                  v60 = (const void **)*((_QWORD *)v59 + 1);
                  v61 = *v59;
                  v84 = v58;
                  if ( v58 )
                  {
                    v70 = v61;
                    v72 = v27;
                    v75 = v60;
                    sub_1623A60((__int64)&v84, v58, 2);
                    v61 = v70;
                    v27 = v72;
                    v60 = v75;
                  }
                  v85 = *(_DWORD *)(a2 + 64);
                  *((_QWORD *)&v67 + 1) = v28 | v27 & 0xFFFFFFFF00000000LL;
                  *(_QWORD *)&v67 = v26;
                  v76 = *((_QWORD *)&v67 + 1);
                  v26 = sub_1D309E0(
                          v81,
                          143,
                          (__int64)&v84,
                          v61,
                          v60,
                          0,
                          *(double *)a4.m128_u64,
                          a5,
                          *(double *)a6.m128i_i64,
                          v67);
                  v28 = v62;
                  v27 = v76;
                  if ( v84 )
                  {
                    sub_161E7C0((__int64)&v84, v84);
                    v27 = v76;
                  }
                  v56 = 1;
                }
                sub_1F99890(a3, a2, v26, v28 | v27 & 0xFFFFFFFF00000000LL, v56);
              }
            }
          }
        }
        else if ( v31 > 658 )
        {
          goto LABEL_59;
        }
        return 0;
      }
      return (__int64 *)sub_217D2A0(a2, (__int64)a3, (__m128i)a4, a5, a6, *(_QWORD *)(a1 + 81552));
    }
    goto LABEL_37;
  }
  if ( v9 == 54 )
  {
LABEL_37:
    if ( v8 <= 0 )
      return 0;
    result = sub_21CD550(a2, (__int64)a3, *(double *)a4.m128_u64, a5, a6);
    if ( !result )
      return 0;
    return result;
  }
  if ( v9 <= 54 )
  {
    if ( v9 != 52 )
      return (__int64 *)sub_217D2A0(a2, (__int64)a3, (__m128i)a4, a5, a6, *(_QWORD *)(a1 + 81552));
    goto LABEL_35;
  }
  if ( (unsigned __int16)(v9 - 57) > 1u )
    return (__int64 *)sub_217D2A0(a2, (__int64)a3, (__m128i)a4, a5, a6, *(_QWORD *)(a1 + 81552));
  if ( v8 <= 1 )
    return 0;
  v13 = *(_QWORD *)(a2 + 72);
  v14 = (__int64 *)a3[2];
  v84 = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)&v84, v13, 2);
    v9 = *(_WORD *)(a2 + 24);
  }
  v85 = *(_DWORD *)(a2 + 64);
  v15 = *(unsigned __int8 **)(a2 + 40);
  v16 = (unsigned int)(v9 != 57) + 55;
  v17 = *(_QWORD *)(a2 + 32);
  v18 = *v15;
  v19 = *(__int64 **)(*(_QWORD *)v17 + 48LL);
  if ( v19 )
  {
    while ( 1 )
    {
      v20 = v19[2];
      if ( *(unsigned __int16 *)(v20 + 24) == (_DWORD)v16 )
      {
        v21 = *(_QWORD *)(v20 + 32);
        if ( *(_QWORD *)v17 == *(_QWORD *)v21
          && *(_DWORD *)(v21 + 8) == *(_DWORD *)(v17 + 8)
          && *(_QWORD *)(v21 + 40) == *(_QWORD *)(v17 + 40)
          && *(_DWORD *)(v21 + 48) == *(_DWORD *)(v17 + 48) )
        {
          break;
        }
      }
      v19 = (__int64 *)v19[4];
      if ( !v19 )
        goto LABEL_51;
    }
    v78 = (const void **)*((_QWORD *)v15 + 1);
    v22 = sub_1D332F0(
            v14,
            v16,
            (__int64)&v84,
            *v15,
            v78,
            0,
            *(double *)a4.m128_u64,
            a5,
            a6,
            *(_QWORD *)v17,
            *(_QWORD *)(v17 + 8),
            *(_OWORD *)(v17 + 40));
    *(_QWORD *)&v24 = sub_1D332F0(
                        v14,
                        54,
                        (__int64)&v84,
                        v18,
                        v78,
                        0,
                        *(double *)a4.m128_u64,
                        a5,
                        a6,
                        (__int64)v22,
                        v23,
                        *(_OWORD *)(v17 + 40));
    v19 = sub_1D332F0(
            v14,
            53,
            (__int64)&v84,
            v18,
            v78,
            0,
            *(double *)a4.m128_u64,
            a5,
            a6,
            *(_QWORD *)v17,
            *(_QWORD *)(v17 + 8),
            v24);
  }
LABEL_51:
  if ( v84 )
    sub_161E7C0((__int64)&v84, v84);
  return v19;
}
