// Function: sub_1DBF150
// Address: 0x1dbf150
//
__int64 __fastcall sub_1DBF150(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        int a7)
{
  __m128i *v11; // rdx
  __int64 result; // rax
  __int64 v13; // r11
  __int64 v14; // r8
  unsigned int v15; // r14d
  _QWORD *v16; // rax
  _QWORD *v17; // rcx
  __int64 v18; // rax
  __int64 i; // rbx
  __int64 v20; // rsi
  unsigned __int64 j; // rax
  __int64 v22; // r9
  __int64 v23; // rsi
  unsigned int v24; // edi
  __int64 *v25; // rcx
  __int64 v26; // r10
  __int64 v27; // rcx
  __int64 v28; // r10
  unsigned __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // r13
  unsigned int v32; // eax
  __int64 v33; // r14
  __int64 v34; // rbx
  unsigned int v35; // r15d
  __int64 v36; // r9
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  bool v42; // cf
  __int64 v43; // r11
  __int64 v44; // rcx
  __int64 v45; // rax
  int v46; // ecx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  int v50; // r13d
  __int128 v51; // [rsp-20h] [rbp-B0h]
  __int64 v52; // [rsp+0h] [rbp-90h]
  __int64 v53; // [rsp+0h] [rbp-90h]
  __int64 v54; // [rsp+0h] [rbp-90h]
  int v55; // [rsp+8h] [rbp-88h]
  __int64 v56; // [rsp+8h] [rbp-88h]
  __int64 v57; // [rsp+8h] [rbp-88h]
  __int64 v58; // [rsp+8h] [rbp-88h]
  __int64 v59; // [rsp+8h] [rbp-88h]
  __int64 v60; // [rsp+10h] [rbp-80h]
  __int64 v61; // [rsp+10h] [rbp-80h]
  __int64 v62; // [rsp+10h] [rbp-80h]
  int v63; // [rsp+10h] [rbp-80h]
  __int64 v64; // [rsp+10h] [rbp-80h]
  __int64 v65; // [rsp+10h] [rbp-80h]
  unsigned __int64 v66; // [rsp+18h] [rbp-78h]
  __int64 v67; // [rsp+20h] [rbp-70h]
  __int64 v68; // [rsp+28h] [rbp-68h]
  unsigned __int64 v71; // [rsp+38h] [rbp-58h]
  __m128i v72; // [rsp+40h] [rbp-50h]
  __m128i v73; // [rsp+40h] [rbp-50h]
  __int64 v74; // [rsp+48h] [rbp-48h]

  v11 = (__m128i *)sub_1DB3C70((__int64 *)a5, a4);
  result = *(_QWORD *)a5;
  if ( v11 != *(__m128i **)a5 )
  {
    result += 24LL * *(unsigned int *)(a5 + 8);
    if ( v11 == (__m128i *)result
      || (result = *(_DWORD *)((v11->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                 | (unsigned int)(v11->m128i_i64[0] >> 1) & 3,
          (unsigned int)result >= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3)) )
    {
      v11 = (__m128i *)((char *)v11 - 24);
      v13 = 0;
    }
    else
    {
      v13 = v11->m128i_i64[1];
    }
    v14 = a5;
    v15 = a6;
    while ( 2 )
    {
      while ( 2 )
      {
        if ( a3 == a2 )
          return result;
LABEL_7:
        v16 = (_QWORD *)(*(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
        v17 = v16;
        if ( !v16 )
          BUG();
        a3 = *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL;
        v18 = *v16;
        if ( (v18 & 4) == 0 && (*((_BYTE *)v17 + 46) & 4) != 0 )
        {
          for ( i = v18; ; i = *(_QWORD *)a3 )
          {
            a3 = i & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(a3 + 46) & 4) == 0 )
              break;
          }
        }
        result = (unsigned int)**(unsigned __int16 **)(a3 + 16) - 12;
        if ( (unsigned __int16)(**(_WORD **)(a3 + 16) - 12) <= 1u )
          continue;
        break;
      }
      v20 = *(_QWORD *)(a1 + 272);
      for ( j = a3; (*(_BYTE *)(j + 46) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
        ;
      v22 = *(_QWORD *)(v20 + 368);
      v23 = *(unsigned int *)(v20 + 384);
      if ( (_DWORD)v23 )
      {
        v24 = (v23 - 1) & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
        v25 = (__int64 *)(v22 + 16LL * v24);
        v26 = *v25;
        if ( j == *v25 )
        {
LABEL_18:
          v27 = v25[1];
          v28 = 0;
          if ( (v11->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            v28 = *(_QWORD *)((v11->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 16);
          v68 = 0;
          v29 = v11->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
          if ( v29 )
            v68 = *(_QWORD *)(v29 + 16);
          v30 = *(_QWORD *)(a3 + 32);
          v71 = v27 & 0xFFFFFFFFFFFFFFF8LL;
          result = 5LL * *(unsigned int *)(a3 + 40);
          v31 = v30 + 40LL * *(unsigned int *)(a3 + 40);
          if ( v30 == v31 )
            continue;
          v32 = v15;
          v66 = a3;
          v33 = a1;
          v34 = v14;
          v67 = v28;
          v35 = v32;
          while ( 1 )
          {
            while ( *(_BYTE *)v30
                 || v35 != *(_DWORD *)(v30 + 8)
                 || (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v33 + 248) + 248LL) + 4LL * ((*(_DWORD *)v30 >> 8) & 0xFFF))
                   & a7) == 0 )
            {
LABEL_34:
              v30 += 40;
              if ( v31 == v30 )
                goto LABEL_43;
            }
            if ( (*(_BYTE *)(v30 + 3) & 0x10) != 0 )
              break;
            if ( !v68 && (v11->m128i_i8[8] & 6) != 0 )
              v11->m128i_i64[1] = v71 | 4;
            if ( (v13 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              v13 = v71 | 4;
            v30 += 40;
            if ( v31 == v30 )
            {
LABEL_43:
              result = v35;
              v14 = v34;
              a1 = v33;
              a3 = v66;
              v15 = result;
              if ( v66 == a2 )
                return result;
              goto LABEL_7;
            }
          }
          v36 = v71 | 4;
          if ( !v67 )
          {
            if ( ((v11->m128i_i8[8] ^ 6) & 6) != 0 )
            {
              v49 = v11[1].m128i_i64[0];
              v11->m128i_i64[0] = v36;
              v13 = 0;
              *(_QWORD *)(v49 + 8) = v36;
              if ( (*(_DWORD *)v30 & 0xFFF00) != 0 )
              {
                if ( (*(_BYTE *)(v30 + 4) & 1) != 0 )
                  v36 = 0;
                v13 = v36;
              }
              goto LABEL_34;
            }
            if ( *(__m128i **)v34 == v11 )
            {
              v57 = v13;
              v73 = _mm_loadu_si128(v11);
              sub_1DB4410(v34, v73.m128i_i64[0], v73.m128i_i64[1], 1);
              v36 = v71 | 4;
              v13 = v57;
            }
            else
            {
              v53 = v13;
              v72 = _mm_loadu_si128(v11);
              v62 = v11[-2].m128i_i64[1];
              sub_1DB4410(v34, v72.m128i_i64[0], v72.m128i_i64[1], 1);
              v36 = v71 | 4;
              v13 = v53;
              if ( (v62 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                v47 = sub_1DB3C70((__int64 *)v34, v62);
                v13 = v53;
                v36 = v71 | 4;
                v11 = (__m128i *)v47;
                if ( (v53 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                  goto LABEL_26;
                goto LABEL_54;
              }
            }
            v11 = *(__m128i **)v34;
          }
          if ( (v13 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
LABEL_26:
            v60 = v13;
            if ( v11->m128i_i64[0] == v36 )
            {
LABEL_31:
              v13 = 0;
              if ( (*(_DWORD *)v30 & 0xFFF00) != 0 && (*(_BYTE *)(v30 + 4) & 1) == 0 )
                v13 = v36;
              goto LABEL_34;
            }
            v37 = 16;
            v52 = v36;
            v55 = *(_DWORD *)(v34 + 72);
            v38 = sub_145CBF0((__int64 *)(v33 + 296), 16, 16);
            v40 = v52;
            v41 = *(unsigned int *)(v34 + 72);
            v42 = (unsigned int)v41 < *(_DWORD *)(v34 + 76);
            *(_DWORD *)v38 = v55;
            v43 = v60;
            *(_QWORD *)(v38 + 8) = v52;
            if ( !v42 )
            {
              v37 = v34 + 80;
              v54 = v60;
              v59 = v38;
              v65 = v40;
              sub_16CD150(v34 + 64, (const void *)(v34 + 80), 0, 8, v39, v40);
              v41 = *(unsigned int *)(v34 + 72);
              v43 = v54;
              v38 = v59;
              v40 = v65;
            }
            v44 = *(_QWORD *)(v34 + 64);
            v74 = v43;
            *(_QWORD *)(v44 + 8 * v41) = v38;
            ++*(_DWORD *)(v34 + 72);
LABEL_30:
            *((_QWORD *)&v51 + 1) = v74;
            *(_QWORD *)&v51 = v40;
            v61 = v40;
            v45 = sub_1DB8610(v34, v37, v41, v44, v39, v40, v51, v38);
            v36 = v61;
            v11 = (__m128i *)v45;
            goto LABEL_31;
          }
LABEL_54:
          v37 = 16;
          v56 = v36;
          v63 = *(_DWORD *)(v34 + 72);
          v38 = sub_145CBF0((__int64 *)(v33 + 296), 16, 16);
          v40 = v56;
          v48 = *(unsigned int *)(v34 + 72);
          *(_DWORD *)v38 = v63;
          *(_QWORD *)(v38 + 8) = v56;
          if ( (unsigned int)v48 >= *(_DWORD *)(v34 + 76) )
          {
            v37 = v34 + 80;
            v58 = v38;
            v64 = v40;
            sub_16CD150(v34 + 64, (const void *)(v34 + 80), 0, 8, v39, v40);
            v48 = *(unsigned int *)(v34 + 72);
            v38 = v58;
            v40 = v64;
          }
          v44 = *(_QWORD *)(v34 + 64);
          *(_QWORD *)(v44 + 8 * v48) = v38;
          ++*(_DWORD *)(v34 + 72);
          v41 = v71 | 6;
          v74 = v71 | 6;
          goto LABEL_30;
        }
        v46 = 1;
        while ( v26 != -8 )
        {
          v50 = v46 + 1;
          v24 = (v23 - 1) & (v46 + v24);
          v25 = (__int64 *)(v22 + 16LL * v24);
          v26 = *v25;
          if ( j == *v25 )
            goto LABEL_18;
          v46 = v50;
        }
      }
      break;
    }
    v25 = (__int64 *)(v22 + 16 * v23);
    goto LABEL_18;
  }
  return result;
}
