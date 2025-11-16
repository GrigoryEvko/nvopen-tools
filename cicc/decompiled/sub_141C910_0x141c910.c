// Function: sub_141C910
// Address: 0x141c910
//
__int64 __fastcall sub_141C910(
        __int64 a1,
        __int64 a2,
        __m128i *a3,
        unsigned __int8 a4,
        __int64 a5,
        __m128i **a6,
        int a7,
        unsigned __int8 a8)
{
  __m128i *v12; // rdi
  __m128i *v13; // rax
  __m128i *v14; // r9
  __int64 v15; // r11
  __m128i *v16; // rbx
  __int64 v17; // rax
  __m128i *v18; // rsi
  __int64 v19; // r9
  int v20; // edx
  unsigned int v21; // esi
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // r12
  __int64 v24; // rdi
  unsigned int v25; // ecx
  unsigned __int64 *v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int64 *v28; // rdx
  __int64 *v29; // r8
  __int64 result; // rax
  __int64 v31; // rsi
  unsigned __int64 v32; // rsi
  _QWORD *v33; // r15
  __int64 v34; // rax
  unsigned __int64 *v35; // rdi
  unsigned int v36; // r10d
  unsigned __int64 *v37; // rsi
  int v38; // eax
  int v39; // esi
  __int64 v40; // r8
  unsigned int v41; // edx
  int v42; // ecx
  unsigned __int64 v43; // rdi
  int v44; // r15d
  unsigned __int64 *v45; // r11
  int v46; // edi
  int v47; // eax
  int v48; // edx
  unsigned __int64 *v49; // r8
  int v50; // r10d
  unsigned int v51; // r14d
  __int64 v52; // rdi
  unsigned __int64 v53; // rsi
  int v54; // r11d
  unsigned __int64 *v55; // r10
  __int64 v56; // [rsp+8h] [rbp-58h]
  __int64 v57; // [rsp+10h] [rbp-50h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  __int64 v60; // [rsp+18h] [rbp-48h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  __m128i v62; // [rsp+20h] [rbp-40h] BYREF

  v12 = *a6;
  v62 = (__m128i)(unsigned __int64)a5;
  v13 = (__m128i *)sub_1411920(v12, (__int64)v12[a7].m128i_i64, (unsigned __int64 *)&v62);
  v16 = v13;
  if ( v13 != v12 && a5 == v13[-1].m128i_i64[0] )
    v16 = v13 - 1;
  if ( v16 != v14 && a5 == v16->m128i_i64[0] )
  {
    v31 = v16->m128i_i64[1];
    if ( (v31 & 7) != 0 )
      return v16->m128i_i64[1];
    v32 = v31 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v32 )
    {
      v56 = v15;
      v33 = (_QWORD *)(v32 + 24);
      sub_1412000(a1 + 128, v32, (4LL * a4) | a3->m128i_i64[0] & 0xFFFFFFFFFFFFFFFBLL);
      v15 = v56;
    }
    else
    {
      v33 = (_QWORD *)(a5 + 40);
    }
    v34 = sub_141C340(a1, a3, a4, v33, a5, v15, 0, a8);
    v16->m128i_i64[1] = v34;
    v19 = v34;
  }
  else
  {
    v17 = sub_141C340(a1, a3, a4, (_QWORD *)(a5 + 40), a5, v15, 0, a8);
    v18 = a6[1];
    v62.m128i_i64[0] = a5;
    v62.m128i_i64[1] = v17;
    v19 = v17;
    if ( v18 != a6[2] )
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(&v62);
        v18 = a6[1];
      }
      v20 = v17 & 7;
      a6[1] = v18 + 1;
      if ( v20 == 2 )
      {
LABEL_10:
        v21 = *(_DWORD *)(a1 + 152);
        v22 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        v23 = (4LL * a4) | a3->m128i_i64[0] & 0xFFFFFFFFFFFFFFFBLL;
        if ( v21 )
        {
          v24 = *(_QWORD *)(a1 + 136);
          v25 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v26 = (unsigned __int64 *)(v24 + 80LL * v25);
          v27 = *v26;
          if ( v22 == *v26 )
          {
LABEL_12:
            v28 = (unsigned __int64 *)v26[2];
            v29 = (__int64 *)(v26 + 1);
            if ( (unsigned __int64 *)v26[3] != v28 )
              goto LABEL_13;
            v35 = &v28[*((unsigned int *)v26 + 9)];
            v36 = *((_DWORD *)v26 + 9);
            if ( v35 != v28 )
            {
              v37 = 0;
              while ( v23 != *v28 )
              {
                if ( *v28 == -2 )
                  v37 = v28;
                if ( v35 == ++v28 )
                {
                  if ( !v37 )
                    goto LABEL_38;
                  *v37 = v23;
                  --*((_DWORD *)v26 + 10);
                  ++v26[1];
                  return v19;
                }
              }
              return v19;
            }
LABEL_38:
            if ( *((_DWORD *)v26 + 8) > v36 )
            {
              *((_DWORD *)v26 + 9) = v36 + 1;
              *v35 = v23;
              ++v26[1];
              return v19;
            }
LABEL_13:
            v59 = v19;
            sub_16CCBA0(v29, v23);
            return v59;
          }
          v44 = 1;
          v45 = 0;
          while ( v27 != -8 )
          {
            if ( !v45 && v27 == -16 )
              v45 = v26;
            v25 = (v21 - 1) & (v44 + v25);
            v26 = (unsigned __int64 *)(v24 + 80LL * v25);
            v27 = *v26;
            if ( v22 == *v26 )
              goto LABEL_12;
            ++v44;
          }
          v46 = *(_DWORD *)(a1 + 144);
          if ( v45 )
            v26 = v45;
          ++*(_QWORD *)(a1 + 128);
          v42 = v46 + 1;
          if ( 4 * (v46 + 1) < 3 * v21 )
          {
            if ( v21 - *(_DWORD *)(a1 + 148) - v42 > v21 >> 3 )
            {
LABEL_35:
              *(_DWORD *)(a1 + 144) = v42;
              if ( *v26 != -8 )
                --*(_DWORD *)(a1 + 148);
              v35 = v26 + 6;
              *v26 = v22;
              v29 = (__int64 *)(v26 + 1);
              v36 = 0;
              v26[1] = 0;
              v26[2] = (unsigned __int64)(v26 + 6);
              v26[3] = (unsigned __int64)(v26 + 6);
              v26[4] = 4;
              *((_DWORD *)v26 + 10) = 0;
              goto LABEL_38;
            }
            v61 = v19;
            sub_1418B70(a1 + 128, v21);
            v47 = *(_DWORD *)(a1 + 152);
            if ( v47 )
            {
              v48 = v47 - 1;
              v49 = 0;
              v19 = v61;
              v50 = 1;
              v51 = (v47 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
              v52 = *(_QWORD *)(a1 + 136);
              v42 = *(_DWORD *)(a1 + 144) + 1;
              v26 = (unsigned __int64 *)(v52 + 80LL * v51);
              v53 = *v26;
              if ( v22 != *v26 )
              {
                while ( v53 != -8 )
                {
                  if ( v53 == -16 && !v49 )
                    v49 = v26;
                  v51 = v48 & (v50 + v51);
                  v26 = (unsigned __int64 *)(v52 + 80LL * v51);
                  v53 = *v26;
                  if ( v22 == *v26 )
                    goto LABEL_35;
                  ++v50;
                }
                if ( v49 )
                  v26 = v49;
              }
              goto LABEL_35;
            }
LABEL_73:
            ++*(_DWORD *)(a1 + 144);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 128);
        }
        v60 = v19;
        sub_1418B70(a1 + 128, 2 * v21);
        v38 = *(_DWORD *)(a1 + 152);
        if ( v38 )
        {
          v39 = v38 - 1;
          v40 = *(_QWORD *)(a1 + 136);
          v19 = v60;
          v41 = (v38 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v42 = *(_DWORD *)(a1 + 144) + 1;
          v26 = (unsigned __int64 *)(v40 + 80LL * v41);
          v43 = *v26;
          if ( v22 != *v26 )
          {
            v54 = 1;
            v55 = 0;
            while ( v43 != -8 )
            {
              if ( v43 == -16 && !v55 )
                v55 = v26;
              v41 = v39 & (v54 + v41);
              v26 = (unsigned __int64 *)(v40 + 80LL * v41);
              v43 = *v26;
              if ( v22 == *v26 )
                goto LABEL_35;
              ++v54;
            }
            if ( v55 )
              v26 = v55;
          }
          goto LABEL_35;
        }
        goto LABEL_73;
      }
      goto LABEL_20;
    }
    v57 = v17;
    sub_1414B00((const __m128i **)a6, v18, &v62);
    v19 = v57;
  }
  v20 = v19 & 7;
  if ( v20 == 2 )
    goto LABEL_10;
LABEL_20:
  result = v19;
  if ( v20 == 1 )
    goto LABEL_10;
  return result;
}
