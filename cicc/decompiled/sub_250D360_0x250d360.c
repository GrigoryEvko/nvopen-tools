// Function: sub_250D360
// Address: 0x250d360
//
__m128i *__fastcall sub_250D360(__int64 a1, __m128i *a2)
{
  __int64 v3; // rax
  __m128i v4; // xmm0
  unsigned __int64 v5; // r13
  unsigned __int8 v6; // cl
  __int64 v7; // kr00_8
  __m128i *result; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned __int64 v35; // rbx
  unsigned __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  unsigned __int64 v45; // r14
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rcx
  __int64 v57; // rbx
  unsigned __int64 i; // r14
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __m128i v73[4]; // [rsp+0h] [rbp-40h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v3 = a2->m128i_i64[0];
  v4 = _mm_loadu_si128(a2);
  *(_DWORD *)(a1 + 8) = 1;
  *(__m128i *)(a1 + 16) = v4;
  v5 = v3 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v3 & 3) == 3 )
    v5 = *(_QWORD *)(v5 + 24);
  if ( *(_BYTE *)v5 > 0x1Cu && (v6 = *(_BYTE *)v5 - 34, v6 <= 0x33u) )
  {
    if ( ((0x8000000000041uLL >> v6) & 1) == 0 )
      v5 = 0;
  }
  else
  {
    v5 = 0;
  }
  v7 = sub_2509800(a2);
  result = (__m128i *)(unsigned __int8)v7;
  switch ( (char)v7 )
  {
    case 2:
    case 6:
      v9 = sub_25096F0(a2);
      goto LABEL_8;
    case 3:
      if ( *(char *)(v5 + 7) >= 0 )
        goto LABEL_44;
      v14 = sub_BD2BC0(v5);
      v16 = v14 + v15;
      v17 = 0;
      if ( *(char *)(v5 + 7) < 0 )
        v17 = sub_BD2BC0(v5);
      if ( (unsigned int)((v16 - v17) >> 4) )
      {
        if ( *(_BYTE *)v5 != 85 )
          goto LABEL_18;
        v45 = *(_QWORD *)(v5 - 32);
        if ( !v45
          || *(_BYTE *)v45
          || *(_QWORD *)(v45 + 24) != *(_QWORD *)(v5 + 80)
          || (*(_BYTE *)(v45 + 33) & 0x20) == 0
          || *(_DWORD *)(v45 + 36) != 11 )
        {
          goto LABEL_18;
        }
      }
      else
      {
LABEL_44:
        v45 = *(_QWORD *)(v5 - 32);
        if ( !v45 || *(_BYTE *)v45 )
          goto LABEL_18;
      }
      sub_250D230((unsigned __int64 *)v73, v45, 2, 0);
      sub_2507260(a1, v73, v46, v47, v48, v49);
      sub_250D230((unsigned __int64 *)v73, v45, 4, 0);
      sub_2507260(a1, v73, v50, v51, v52, v53);
      if ( (*(_BYTE *)(v45 + 2) & 1) != 0 )
      {
        sub_B2C6D0(v45, (__int64)v73, v54, v55);
        v56 = *(_QWORD *)(v45 + 96);
        v57 = v56 + 40LL * *(_QWORD *)(v45 + 104);
        if ( (*(_BYTE *)(v45 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v45, (__int64)v73, 5LL * *(_QWORD *)(v45 + 104), v56);
          v56 = *(_QWORD *)(v45 + 96);
        }
      }
      else
      {
        v56 = *(_QWORD *)(v45 + 96);
        v57 = v56 + 40LL * *(_QWORD *)(v45 + 104);
      }
      for ( i = v56; v57 != i; i += 40LL )
      {
        if ( (unsigned __int8)sub_B2D750(i) )
        {
          v59 = *(unsigned int *)(i + 32);
          if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
            v60 = *(_QWORD *)(v5 - 8);
          else
            v60 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
          v73[0].m128i_i64[1] = 0;
          v73[0].m128i_i64[0] = (v60 + 32 * v59) | 3;
          nullsub_1518();
          sub_2507260(a1, v73, v61, v62, v63, v64);
          v73[0].m128i_i64[0] = sub_250D2C0(
                                  *(_QWORD *)(v5
                                            + 32
                                            * (*(unsigned int *)(i + 32)
                                             - (unsigned __int64)(*(_DWORD *)(v5 + 4) & 0x7FFFFFF))),
                                  0);
          v73[0].m128i_i64[1] = v65;
          sub_2507260(a1, v73, v65, v66, v67, v68);
          sub_250D230((unsigned __int64 *)v73, i, 6, 0);
          sub_2507260(a1, v73, v69, v70, v71, v72);
        }
      }
LABEL_18:
      sub_250D230((unsigned __int64 *)v73, v5, 5, 0);
      return sub_2507260(a1, v73, v18, v19, v20, v21);
    case 5:
      if ( *(char *)(v5 + 7) >= 0 )
        goto LABEL_41;
      v22 = sub_BD2BC0(v5);
      v24 = v22 + v23;
      v25 = 0;
      if ( *(char *)(v5 + 7) < 0 )
        v25 = sub_BD2BC0(v5);
      result = (__m128i *)((v24 - v25) >> 4);
      if ( (_DWORD)result )
      {
        if ( *(_BYTE *)v5 == 85 )
        {
          v9 = *(_QWORD *)(v5 - 32);
          if ( v9 )
          {
            if ( !*(_BYTE *)v9 )
            {
              result = *(__m128i **)(v5 + 80);
              if ( *(__m128i **)(v9 + 24) == result && (*(_BYTE *)(v9 + 33) & 0x20) != 0 && *(_DWORD *)(v9 + 36) == 11 )
              {
LABEL_8:
                sub_250D230((unsigned __int64 *)v73, v9, 4, 0);
                return sub_2507260(a1, v73, v10, v11, v12, v13);
              }
            }
          }
        }
      }
      else
      {
LABEL_41:
        v9 = *(_QWORD *)(v5 - 32);
        if ( v9 && !*(_BYTE *)v9 )
          goto LABEL_8;
      }
      return result;
    case 7:
      if ( *(char *)(v5 + 7) >= 0 )
        goto LABEL_36;
      v26 = sub_BD2BC0(v5);
      v28 = v26 + v27;
      v29 = 0;
      if ( *(char *)(v5 + 7) < 0 )
        v29 = sub_BD2BC0(v5);
      if ( (unsigned int)((v28 - v29) >> 4) )
      {
        if ( *(_BYTE *)v5 != 85 )
          goto LABEL_35;
        v35 = *(_QWORD *)(v5 - 32);
        if ( !v35
          || *(_BYTE *)v35
          || *(_QWORD *)(v35 + 24) != *(_QWORD *)(v5 + 80)
          || (*(_BYTE *)(v35 + 33) & 0x20) == 0
          || *(_DWORD *)(v35 + 36) != 11 )
        {
          goto LABEL_35;
        }
      }
      else
      {
LABEL_36:
        v35 = *(_QWORD *)(v5 - 32);
        if ( !v35 || *(_BYTE *)v35 )
          goto LABEL_35;
      }
      v36 = sub_250C680(a2->m128i_i64);
      if ( v36 )
      {
        sub_250D230((unsigned __int64 *)v73, v36, 6, 0);
        sub_2507260(a1, v73, v37, v38, v39, v40);
      }
      sub_250D230((unsigned __int64 *)v73, v35, 4, 0);
      sub_2507260(a1, v73, v41, v42, v43, v44);
LABEL_35:
      v30 = sub_250D070(a2);
      v73[0].m128i_i64[0] = sub_250D2C0(v30, 0);
      v73[0].m128i_i64[1] = v31;
      return sub_2507260(a1, v73, v31, v32, v33, v34);
    default:
      return (__m128i *)v7;
  }
}
