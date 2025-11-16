// Function: sub_621760
// Address: 0x621760
//
__int64 __fastcall sub_621760(const __m128i *a1, const __m128i *a2, __int16 *a3, __int16 *a4, int a5, _BOOL4 *a6)
{
  __m128i *v6; // r13
  __m128i v7; // xmm0
  __int64 result; // rax
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // esi
  __int64 v14; // rax
  int v15; // r10d
  int v16; // edi
  __m128i *v17; // r8
  unsigned __int64 v18; // r11
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __m128i *v21; // r9
  unsigned __int64 v22; // rax
  __m128i *v23; // r8
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __m128i *v26; // r9
  unsigned __int64 v27; // rax
  int v28; // edx
  _QWORD *v29; // rbx
  unsigned __int16 *v30; // r9
  unsigned int v31; // r10d
  char *v32; // r14
  __int64 v33; // rdx
  unsigned __int64 v34; // rax
  unsigned __int16 v35; // r11
  unsigned __int16 *v36; // rsi
  unsigned __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int16 *v39; // rdi
  unsigned __int64 v40; // rdx
  int v41; // edx
  __int64 i; // rcx
  int v43; // esi
  __m128i v44; // xmm4
  unsigned int v45; // ecx
  __int64 v46; // rsi
  __int64 v47; // rcx
  _BOOL8 v48; // rsi
  __int64 v49; // rdx
  __m128i *v50; // rcx
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  __m128i v53; // xmm2
  unsigned __int64 v54; // [rsp+0h] [rbp-150h]
  unsigned __int64 v56; // [rsp+20h] [rbp-130h]
  int v57; // [rsp+28h] [rbp-128h]
  int v58; // [rsp+2Ch] [rbp-124h]
  __int64 v62; // [rsp+48h] [rbp-108h]
  char *v63; // [rsp+58h] [rbp-F8h]
  __int128 v64; // [rsp+60h] [rbp-F0h]
  __m128i v65; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v66; // [rsp+80h] [rbp-D0h] BYREF
  __m128i v67; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v68; // [rsp+A0h] [rbp-B0h] BYREF
  __int16 v69; // [rsp+B0h] [rbp-A0h]
  __m128i v70[2]; // [rsp+C0h] [rbp-90h] BYREF
  __int16 v71; // [rsp+E0h] [rbp-70h] BYREF
  __int128 v72; // [rsp+F0h] [rbp-60h] BYREF
  __int128 v73; // [rsp+100h] [rbp-50h] BYREF
  __int16 v74; // [rsp+110h] [rbp-40h]

  v71 = 0;
  v74 = 0;
  v69 = 0;
  memset(v70, 0, sizeof(v70));
  v72 = 0;
  v73 = 0;
  v68 = 0;
  v65 = _mm_loadu_si128(a2);
  if ( a5 )
  {
    v6 = (__m128i *)a1;
    v58 = 0;
    if ( a1->m128i_i16[0] < 0 )
    {
      v6 = &v66;
      v66 = _mm_loadu_si128(a1);
      sub_621710(v66.m128i_i16, a6);
      v58 = 1;
    }
    v57 = v58;
    if ( v65.m128i_i16[0] < 0 )
    {
      sub_621710(v65.m128i_i16, a6);
      v57 = v58 ^ 1;
    }
  }
  else
  {
    v58 = 0;
    v6 = (__m128i *)a1;
    v57 = 0;
  }
  sub_620D80(&v67, 0);
  if ( (unsigned int)sub_621000(v65.m128i_i16, 0, v67.m128i_i16, 0) )
  {
    result = sub_621000(v6->m128i_i16, 0, v67.m128i_i16, 0);
    if ( (_DWORD)result )
    {
      if ( (int)sub_621000(v6->m128i_i16, 0, v65.m128i_i16, 0) < 0 )
      {
        *(__m128i *)a3 = _mm_load_si128(&v67);
        v44 = _mm_loadu_si128(a1);
        *(__m128i *)a4 = v44;
        result = 0;
        *((_QWORD *)&v64 + 1) = v44.m128i_i64[1];
      }
      else
      {
        v10 = 0;
        *(__m128i *)((char *)v70 + 2) = _mm_loadu_si128(v6);
        while ( 1 )
        {
          v11 = v65.m128i_u16[v10];
          v12 = v10;
          if ( (_WORD)v11 )
            break;
          if ( ++v10 == 8 )
          {
            v11 = v65.m128i_u16[0];
            v13 = 0;
            v12 = 0;
            goto LABEL_17;
          }
        }
        v13 = 8 - v10;
LABEL_17:
        v14 = 0;
        while ( !v70[0].m128i_i16[v14] )
        {
          if ( ++v14 == 9 )
          {
            v15 = -1;
            v16 = 0;
            goto LABEL_21;
          }
        }
        v15 = v14 - 1;
        v16 = 9 - v14;
LABEL_21:
        v17 = (__m128i *)((char *)&v65.m128i_u64[1] + 6);
        v18 = (unsigned __int16)(0x10000uLL / (v11 + 1));
        v19 = 0;
        v54 = v18;
        do
        {
          v20 = v17->m128i_u16[0];
          v21 = v17;
          v17 = (__m128i *)((char *)v17 - 2);
          v22 = v18 * v20 + v19;
          v17->m128i_i16[1] = v22;
          v19 = v22 >> 16;
        }
        while ( &v65 != v21 );
        v23 = (__m128i *)&v71;
        v24 = 0;
        do
        {
          v25 = v23->m128i_u16[0];
          v26 = v23;
          v23 = (__m128i *)((char *)v23 - 2);
          v27 = v18 * v25 + v24;
          v23->m128i_i16[1] = v27;
          v24 = v27 >> 16;
        }
        while ( v70 != v26 );
        v56 = v65.m128i_u16[v12];
        v28 = v13 - v16 + 7;
        if ( v28 <= 7 )
        {
          v29 = (__int64 *)((char *)v65.m128i_i64 + 2 * v12);
          v62 = v28;
          v30 = (unsigned __int16 *)v70 + v15;
          v31 = 2 * (8 - v12);
          if ( 8 - v12 <= 0 )
            v31 = 2;
          v32 = (char *)v29 + v31;
          v63 = (char *)&v72 + v31 + 2;
          do
          {
            v33 = *v30;
            LOWORD(v34) = -1;
            v35 = *v30;
            if ( v56 != v33 )
              v34 = ((v33 << 16) + (unsigned __int64)v30[1]) / v56;
            while ( 1 )
            {
              v74 = 0;
              v72 = 0;
              v73 = 0;
              if ( v31 >= 8 )
              {
                *(_QWORD *)((char *)&v72 + 2) = *v29;
                *((_QWORD *)v63 - 1) = *((_QWORD *)v32 - 1);
                if ( ((v31 + (unsigned int)((char *)&v72 + 2 - ((char *)&v72 + 8))) & 0xFFFFFFF8) >= 8 )
                {
                  v45 = 0;
                  do
                  {
                    v46 = v45;
                    v45 += 8;
                    *(_QWORD *)((char *)&v72 + v46 + 8) = *(_QWORD *)((char *)v29
                                                                    + v46
                                                                    - ((char *)&v72
                                                                     + 2
                                                                     - ((char *)&v72
                                                                      + 8)));
                  }
                  while ( v45 < ((v31 + (unsigned int)((char *)&v72 + 2 - ((char *)&v72 + 8))) & 0xFFFFFFF8) );
                }
              }
              else if ( (v31 & 4) != 0 )
              {
                *(_DWORD *)((char *)&v72 + 2) = *(_DWORD *)v29;
                *((_DWORD *)v63 - 1) = *((_DWORD *)v32 - 1);
              }
              else if ( v31 )
              {
                BYTE2(v72) = *(_BYTE *)v29;
                if ( (v31 & 2) != 0 )
                  *((_WORD *)v63 - 1) = *((_WORD *)v32 - 1);
              }
              v36 = (unsigned __int16 *)&v73;
              v37 = 0;
              do
              {
                v38 = *v36;
                v39 = v36--;
                v40 = (unsigned __int16)v34 * v38 + v37;
                v36[1] = v40;
                v37 = v40 >> 16;
              }
              while ( v39 != (unsigned __int16 *)&v72 );
              v41 = v35;
              for ( i = 0; ; v41 = v30[i] )
              {
                v43 = *(unsigned __int16 *)((char *)&v72 + i * 2);
                if ( v41 != v43 )
                  break;
                if ( ++i == 9 )
                  goto LABEL_49;
              }
              if ( v41 - v43 >= 0 )
                break;
              LOWORD(v34) = v34 - 1;
            }
LABEL_49:
            v47 = 8;
            v48 = 0;
            do
            {
              v49 = v30[v47] - (unsigned __int64)*(unsigned __int16 *)((char *)&v72 + v47 * 2) - v48;
              v48 = v49 < 0;
              v30[v47--] = v49;
            }
            while ( v47 != -1 );
            ++v30;
            v68.m128i_i16[v62++] = v34;
          }
          while ( (int)v62 <= 7 );
        }
        v50 = (__m128i *)&v70[0].m128i_i16[1];
        v51 = 0;
        do
        {
          v52 = v50->m128i_u16[0];
          v50 = (__m128i *)((char *)v50 + 2);
          v50[-1].m128i_i16[7] = (v51 + v52) / v54;
          v51 = ((v51 + v52) % v54) << 16;
        }
        while ( &v70[1].m128i_i16[1] != (__int16 *)v50 );
        v53 = _mm_loadu_si128((const __m128i *)&v70[0].m128i_i16[1]);
        *(__m128i *)a3 = _mm_load_si128(&v68);
        *(__m128i *)a4 = v53;
        if ( v58 )
          sub_621710(a4, a6);
        if ( v57 )
        {
          sub_621710(a3, a6);
          result = 0;
        }
        else
        {
          result = 0;
          if ( a5 )
            result = (unsigned int)*a3 >> 31;
        }
      }
    }
    else
    {
      v9 = _mm_load_si128(&v67);
      *(__m128i *)a3 = v9;
      *(__m128i *)a4 = v9;
    }
  }
  else
  {
    v7 = _mm_load_si128(&v67);
    *(__m128i *)a3 = v7;
    *(__m128i *)a4 = v7;
    result = 1;
  }
  *a6 = result;
  return result;
}
