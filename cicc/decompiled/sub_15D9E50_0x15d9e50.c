// Function: sub_15D9E50
// Address: 0x15d9e50
//
__int64 __fastcall sub_15D9E50(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 i; // r15
  __int64 v8; // rax
  char v9; // al
  __int64 *v10; // rsi
  int v11; // eax
  __int64 v12; // rbx
  const __m128i *v13; // r14
  unsigned __int64 v14; // rax
  char v15; // al
  __int64 *v16; // rsi
  const __m128i *v17; // r8
  unsigned int v18; // eax
  int v19; // eax
  unsigned int v20; // r9d
  unsigned __int64 v21; // rax
  unsigned int v22; // eax
  int v23; // eax
  unsigned int v24; // r8d
  unsigned __int64 v25; // rax
  unsigned int v26; // esi
  unsigned int v27; // esi
  __m128i *v28; // rcx
  __int64 v29; // r14
  char v30; // al
  __int64 *v31; // rdx
  __m128i *v32; // rbx
  const __m128i *v33; // r15
  __int64 v34; // rax
  char v35; // al
  __int64 *v36; // rdx
  unsigned int v37; // eax
  int v38; // ecx
  unsigned int v39; // esi
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  unsigned int v42; // eax
  int v43; // ecx
  unsigned int v44; // esi
  unsigned __int64 v45; // rax
  __int64 v48; // [rsp+8h] [rbp-A8h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  int v53; // [rsp+30h] [rbp-80h]
  const __m128i *v54; // [rsp+30h] [rbp-80h]
  int v55; // [rsp+30h] [rbp-80h]
  __int64 *v56; // [rsp+58h] [rbp-58h] BYREF
  __int64 v57; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v58; // [rsp+68h] [rbp-48h]
  __int64 *v59; // [rsp+70h] [rbp-40h] BYREF
  unsigned __int64 v60; // [rsp+78h] [rbp-38h]

  v48 = a3 & 1;
  v50 = (a3 - 1) / 2;
  if ( a2 < v50 )
  {
    for ( i = a2; ; i = v12 )
    {
      v12 = 2 * (i + 1);
      v13 = (const __m128i *)(a1 + 32 * (i + 1));
      v14 = v13->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v57 = v13->m128i_i64[0];
      v58 = v14;
      v15 = sub_15D0A10(a6, &v57, &v59);
      v16 = v59;
      v17 = v13 - 1;
      if ( v15 )
      {
        v53 = *((_DWORD *)v59 + 4);
        goto LABEL_4;
      }
      v18 = *(_DWORD *)(a6 + 8);
      ++*(_QWORD *)a6;
      v19 = (v18 >> 1) + 1;
      if ( (*(_BYTE *)(a6 + 8) & 1) != 0 )
      {
        v20 = 4;
        if ( (unsigned int)(4 * v19) >= 0xC )
        {
LABEL_24:
          v54 = v13 - 1;
          v26 = 2 * v20;
          goto LABEL_25;
        }
      }
      else
      {
        v20 = *(_DWORD *)(a6 + 24);
        if ( 4 * v19 >= 3 * v20 )
          goto LABEL_24;
      }
      if ( v20 - (v19 + *(_DWORD *)(a6 + 12)) > v20 >> 3 )
        goto LABEL_14;
      v54 = v13 - 1;
      v26 = v20;
LABEL_25:
      sub_15D0B40(a6, v26);
      sub_15D0A10(a6, &v57, &v59);
      v16 = v59;
      v17 = v54;
      v19 = (*(_DWORD *)(a6 + 8) >> 1) + 1;
LABEL_14:
      *(_DWORD *)(a6 + 8) = *(_DWORD *)(a6 + 8) & 1 | (2 * v19);
      if ( *v16 != -8 || v16[1] != -8 )
        --*(_DWORD *)(a6 + 12);
      v53 = 0;
      *v16 = v57;
      v21 = v58;
      *((_DWORD *)v16 + 4) = 0;
      v16[1] = v21;
LABEL_4:
      v8 = v17->m128i_i64[1];
      v59 = (__int64 *)v17->m128i_i64[0];
      v60 = v8 & 0xFFFFFFFFFFFFFFF8LL;
      v9 = sub_15D0A10(a6, (__int64 *)&v59, &v56);
      v10 = v56;
      if ( v9 )
      {
        v11 = *((_DWORD *)v56 + 4);
        goto LABEL_6;
      }
      v22 = *(_DWORD *)(a6 + 8);
      ++*(_QWORD *)a6;
      v23 = (v22 >> 1) + 1;
      if ( (*(_BYTE *)(a6 + 8) & 1) != 0 )
      {
        v24 = 4;
        if ( (unsigned int)(4 * v23) >= 0xC )
        {
LABEL_27:
          v27 = 2 * v24;
          goto LABEL_28;
        }
      }
      else
      {
        v24 = *(_DWORD *)(a6 + 24);
        if ( 4 * v23 >= 3 * v24 )
          goto LABEL_27;
      }
      if ( v24 - (v23 + *(_DWORD *)(a6 + 12)) > v24 >> 3 )
        goto LABEL_20;
      v27 = v24;
LABEL_28:
      sub_15D0B40(a6, v27);
      sub_15D0A10(a6, (__int64 *)&v59, &v56);
      v10 = v56;
      v23 = (*(_DWORD *)(a6 + 8) >> 1) + 1;
LABEL_20:
      *(_DWORD *)(a6 + 8) = *(_DWORD *)(a6 + 8) & 1 | (2 * v23);
      if ( *v10 != -8 || v10[1] != -8 )
        --*(_DWORD *)(a6 + 12);
      *v10 = (__int64)v59;
      v25 = v60;
      *((_DWORD *)v10 + 4) = 0;
      v10[1] = v25;
      v11 = 0;
LABEL_6:
      if ( v53 > v11 )
      {
        --v12;
        v13 = (const __m128i *)(a1 + 16 * v12);
      }
      *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(v13);
      if ( v12 >= v50 )
      {
        v28 = (__m128i *)v13;
        if ( v48 )
          goto LABEL_34;
        goto LABEL_49;
      }
    }
  }
  v12 = a2;
  v28 = (__m128i *)(a1 + 16 * a2);
  if ( (a3 & 1) == 0 )
  {
LABEL_49:
    if ( (a3 - 2) / 2 == v12 )
    {
      v41 = v12 + 1;
      v12 = 2 * (v12 + 1) - 1;
      *v28 = _mm_loadu_si128((const __m128i *)(a1 + 32 * v41 - 16));
      v28 = (__m128i *)(a1 + 16 * v12);
    }
LABEL_34:
    v29 = (v12 - 1) / 2;
    if ( v12 > a2 )
    {
      while ( 1 )
      {
        v33 = (const __m128i *)(a1 + 16 * v29);
        v34 = v33->m128i_i64[1];
        v57 = v33->m128i_i64[0];
        v58 = v34 & 0xFFFFFFFFFFFFFFF8LL;
        v35 = sub_15D0A10(a6, &v57, &v59);
        v36 = v59;
        if ( v35 )
        {
          v55 = *((_DWORD *)v59 + 4);
          goto LABEL_37;
        }
        v37 = *(_DWORD *)(a6 + 8);
        ++*(_QWORD *)a6;
        v38 = (v37 >> 1) + 1;
        if ( (*(_BYTE *)(a6 + 8) & 1) != 0 )
        {
          v39 = 4;
          if ( (unsigned int)(4 * v38) < 0xC )
          {
LABEL_44:
            if ( v39 - (*(_DWORD *)(a6 + 12) + v38) > v39 >> 3 )
              goto LABEL_45;
            goto LABEL_61;
          }
        }
        else
        {
          v39 = *(_DWORD *)(a6 + 24);
          if ( 4 * v38 < 3 * v39 )
            goto LABEL_44;
        }
        v39 *= 2;
LABEL_61:
        sub_15D0B40(a6, v39);
        sub_15D0A10(a6, &v57, &v59);
        v36 = v59;
        v37 = *(_DWORD *)(a6 + 8);
LABEL_45:
        *(_DWORD *)(a6 + 8) = (2 * (v37 >> 1) + 2) | v37 & 1;
        if ( *v36 != -8 || v36[1] != -8 )
          --*(_DWORD *)(a6 + 12);
        v55 = 0;
        *v36 = v57;
        v40 = v58;
        *((_DWORD *)v36 + 4) = 0;
        v36[1] = v40;
LABEL_37:
        v59 = a4;
        v60 = a5 & 0xFFFFFFFFFFFFFFF8LL;
        v30 = sub_15D0A10(a6, (__int64 *)&v59, &v56);
        v31 = v56;
        if ( v30 )
        {
          v32 = (__m128i *)(a1 + 16 * v12);
          if ( v55 <= *((_DWORD *)v56 + 4) )
            goto LABEL_57;
          goto LABEL_39;
        }
        v42 = *(_DWORD *)(a6 + 8);
        ++*(_QWORD *)a6;
        v43 = (v42 >> 1) + 1;
        if ( (*(_BYTE *)(a6 + 8) & 1) != 0 )
        {
          v44 = 4;
          if ( (unsigned int)(4 * v43) >= 0xC )
          {
LABEL_63:
            v44 *= 2;
LABEL_64:
            sub_15D0B40(a6, v44);
            sub_15D0A10(a6, (__int64 *)&v59, &v56);
            v31 = v56;
            v42 = *(_DWORD *)(a6 + 8);
            goto LABEL_54;
          }
        }
        else
        {
          v44 = *(_DWORD *)(a6 + 24);
          if ( 4 * v43 >= 3 * v44 )
            goto LABEL_63;
        }
        if ( v44 - (*(_DWORD *)(a6 + 12) + v43) <= v44 >> 3 )
          goto LABEL_64;
LABEL_54:
        *(_DWORD *)(a6 + 8) = (2 * (v42 >> 1) + 2) | v42 & 1;
        if ( *v31 != -8 || v31[1] != -8 )
          --*(_DWORD *)(a6 + 12);
        v32 = (__m128i *)(a1 + 16 * v12);
        *v31 = (__int64)v59;
        v45 = v60;
        *((_DWORD *)v31 + 4) = 0;
        v31[1] = v45;
        if ( v55 <= 0 )
        {
LABEL_57:
          v28 = v32;
          break;
        }
LABEL_39:
        *v32 = _mm_loadu_si128(v33);
        v12 = v29;
        if ( a2 >= v29 )
        {
          v28 = (__m128i *)(a1 + 16 * v29);
          break;
        }
        v29 = (v29 - 1) / 2;
      }
    }
  }
  v28->m128i_i64[0] = (__int64)a4;
  v28->m128i_i64[1] = a5;
  return a5;
}
