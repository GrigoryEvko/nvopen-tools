// Function: sub_200F8F0
// Address: 0x200f8f0
//
__int64 __fastcall sub_200F8F0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  int v3; // r14d
  char v6; // cl
  __int64 v7; // rdi
  int v8; // esi
  int v9; // r9d
  unsigned int i; // eax
  __int64 v11; // r13
  __int64 v12; // r8
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r8
  __int64 result; // rax
  __int64 v18; // rsi
  int v19; // r13d
  __int64 v20; // r9
  unsigned int v21; // ecx
  __int64 v22; // r15
  unsigned int v23; // ecx
  char v24; // r9
  int v25; // r9d
  __int64 v26; // r10
  int v27; // r8d
  unsigned int v28; // ecx
  _DWORD *v29; // rdi
  int v30; // esi
  unsigned int v31; // esi
  unsigned int v32; // ecx
  int v33; // edi
  unsigned int v34; // ecx
  _DWORD *v35; // r11
  int v36; // edi
  unsigned int v37; // r9d
  __m128i v38; // xmm0
  int v39; // r13d
  __int64 v40; // r8
  int v41; // edi
  int v42; // r11d
  __int64 v43; // r10
  unsigned int j; // ecx
  unsigned int v45; // ecx
  __int64 v46; // r8
  int v47; // ecx
  unsigned int v48; // esi
  int v49; // edi
  __int64 v50; // rdi
  int v51; // r8d
  int v52; // r11d
  unsigned int k; // ecx
  unsigned int v54; // ecx
  __int64 v55; // r8
  int v56; // esi
  unsigned int v57; // edi
  int v58; // ecx
  int v59; // r10d
  _DWORD *v60; // r9
  int v61; // r11d
  int v62; // ecx
  int v63; // edi
  int v64; // ecx
  int v65; // esi
  int v66; // r10d
  int v67; // esi
  int v68; // esi
  __int64 v69; // [rsp+0h] [rbp-60h]
  int v70; // [rsp+0h] [rbp-60h]
  __int64 v71; // [rsp+0h] [rbp-60h]
  int v72; // [rsp+0h] [rbp-60h]
  int v73; // [rsp+Ch] [rbp-54h]
  int v74; // [rsp+Ch] [rbp-54h]
  int v75; // [rsp+Ch] [rbp-54h]
  int v76; // [rsp+Ch] [rbp-54h]
  __m128i v77; // [rsp+18h] [rbp-48h] BYREF

  v3 = a3;
  v6 = *(_BYTE *)(a1 + 144) & 1;
  if ( v6 )
  {
    v7 = a1 + 152;
    v8 = 7;
  }
  else
  {
    v14 = *(unsigned int *)(a1 + 160);
    v7 = *(_QWORD *)(a1 + 152);
    if ( !(_DWORD)v14 )
      goto LABEL_10;
    v8 = v14 - 1;
  }
  v9 = 1;
  for ( i = v8 & (a3 + ((a2 >> 9) ^ (a2 >> 4))); ; i = v8 & v13 )
  {
    v11 = v7 + 24LL * i;
    v12 = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 != a2 )
      break;
    if ( *(_DWORD *)(v11 + 8) == (_DWORD)a3 )
      goto LABEL_12;
    if ( !v12 )
      goto LABEL_23;
LABEL_6:
    v13 = v9 + i;
    ++v9;
  }
  if ( v12 )
    goto LABEL_6;
LABEL_23:
  if ( *(_DWORD *)(v11 + 8) != -1 )
    goto LABEL_6;
  if ( v6 )
  {
    v15 = 192;
    goto LABEL_11;
  }
  v14 = *(unsigned int *)(a1 + 160);
LABEL_10:
  v15 = 24 * v14;
LABEL_11:
  v11 = v7 + v15;
LABEL_12:
  if ( v6 )
  {
    v16 = *(_QWORD *)(a1 + 136);
    if ( v11 != v7 + 192 )
    {
LABEL_14:
      sub_200D1B0(a1, (int *)(v11 + 16));
      return *(unsigned int *)(v11 + 16);
    }
    result = *(unsigned int *)(a1 + 132);
    LODWORD(v18) = 8;
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 136);
    v18 = *(unsigned int *)(a1 + 160);
    if ( v11 != v7 + 24 * v18 )
      goto LABEL_14;
    result = *(unsigned int *)(a1 + 132);
    if ( !*(_DWORD *)(a1 + 160) )
    {
      v32 = *(_DWORD *)(a1 + 144);
      v20 = 0;
      *(_QWORD *)(a1 + 136) = v16 + 1;
      v33 = (v32 >> 1) + 1;
      goto LABEL_35;
    }
  }
  v19 = 1;
  v20 = 0;
  v21 = (v18 - 1) & (a3 + ((a2 >> 9) ^ (a2 >> 4)));
  while ( 2 )
  {
    v22 = v7 + 24LL * v21;
    if ( *(_QWORD *)v22 == a2 && *(_DWORD *)(v22 + 8) == (_DWORD)a3 )
      goto LABEL_28;
    if ( *(_QWORD *)v22 )
    {
LABEL_20:
      v23 = v19 + v21;
      ++v19;
      v21 = (v18 - 1) & v23;
      continue;
    }
    break;
  }
  v61 = *(_DWORD *)(v22 + 8);
  if ( v61 != -1 )
  {
    if ( !v20 && v61 == -2 )
      v20 = v7 + 24LL * v21;
    goto LABEL_20;
  }
  v32 = *(_DWORD *)(a1 + 144);
  if ( !v20 )
    v20 = v22;
  *(_QWORD *)(a1 + 136) = v16 + 1;
  v33 = (v32 >> 1) + 1;
LABEL_35:
  if ( 4 * v33 >= (unsigned int)(3 * v18) )
  {
    v69 = a3;
    v73 = result;
    sub_200F070((const __m128i *)(a1 + 136), 2 * v18);
    LODWORD(result) = v73;
    a3 = v69;
    if ( (*(_BYTE *)(a1 + 144) & 1) != 0 )
    {
      v40 = a1 + 152;
      v41 = 7;
    }
    else
    {
      v63 = *(_DWORD *)(a1 + 160);
      v40 = *(_QWORD *)(a1 + 152);
      if ( !v63 )
        goto LABEL_128;
      v41 = v63 - 1;
    }
    v42 = 1;
    v43 = 0;
    for ( j = v41 & (v69 + ((a2 >> 9) ^ (a2 >> 4))); ; j = v41 & v45 )
    {
      v20 = v40 + 24LL * j;
      if ( *(_QWORD *)v20 == a2 && *(_DWORD *)(v20 + 8) == v3 )
        break;
      if ( !*(_QWORD *)v20 )
      {
        v68 = *(_DWORD *)(v20 + 8);
        if ( v68 == -1 )
        {
LABEL_120:
          if ( v43 )
            v20 = v43;
          goto LABEL_99;
        }
        if ( v68 == -2 && !v43 )
          v43 = v40 + 24LL * j;
      }
      v45 = v42 + j;
      ++v42;
    }
    goto LABEL_99;
  }
  if ( (int)v18 - *(_DWORD *)(a1 + 148) - v33 > (unsigned int)v18 >> 3 )
    goto LABEL_37;
  v71 = a3;
  v75 = result;
  sub_200F070((const __m128i *)(a1 + 136), v18);
  LODWORD(result) = v75;
  a3 = v71;
  if ( (*(_BYTE *)(a1 + 144) & 1) == 0 )
  {
    v64 = *(_DWORD *)(a1 + 160);
    v50 = *(_QWORD *)(a1 + 152);
    if ( v64 )
    {
      v51 = v64 - 1;
      goto LABEL_67;
    }
LABEL_128:
    *(_DWORD *)(a1 + 144) = (2 * (*(_DWORD *)(a1 + 144) >> 1) + 2) | *(_DWORD *)(a1 + 144) & 1;
    BUG();
  }
  v50 = a1 + 152;
  v51 = 7;
LABEL_67:
  v52 = 1;
  v43 = 0;
  for ( k = v51 & (v71 + ((a2 >> 9) ^ (a2 >> 4))); ; k = v51 & v54 )
  {
    v20 = v50 + 24LL * k;
    if ( *(_QWORD *)v20 == a2 && *(_DWORD *)(v20 + 8) == v3 )
      break;
    if ( !*(_QWORD *)v20 )
    {
      v67 = *(_DWORD *)(v20 + 8);
      if ( v67 == -1 )
        goto LABEL_120;
      if ( v67 == -2 && !v43 )
        v43 = v50 + 24LL * k;
    }
    v54 = v52 + k;
    ++v52;
  }
LABEL_99:
  v32 = *(_DWORD *)(a1 + 144);
LABEL_37:
  *(_DWORD *)(a1 + 144) = (2 * (v32 >> 1) + 2) | v32 & 1;
  if ( *(_QWORD *)v20 || *(_DWORD *)(v20 + 8) != -1 )
    --*(_DWORD *)(a1 + 148);
  *(_QWORD *)v20 = a2;
  *(_DWORD *)(v20 + 8) = a3;
  *(_DWORD *)(v20 + 16) = result;
  result = *(unsigned int *)(a1 + 132);
LABEL_28:
  v24 = *(_BYTE *)(a1 + 352);
  v77.m128i_i64[0] = a2;
  v77.m128i_i64[1] = a3;
  v25 = v24 & 1;
  if ( v25 )
  {
    v26 = a1 + 360;
    v27 = 7;
    goto LABEL_30;
  }
  v31 = *(_DWORD *)(a1 + 368);
  v26 = *(_QWORD *)(a1 + 360);
  if ( !v31 )
  {
    v34 = *(_DWORD *)(a1 + 352);
    v35 = 0;
    ++*(_QWORD *)(a1 + 344);
    v36 = (v34 >> 1) + 1;
    goto LABEL_41;
  }
  v27 = v31 - 1;
LABEL_30:
  v28 = v27 & (37 * result);
  v29 = (_DWORD *)(v26 + 24LL * v28);
  v30 = *v29;
  if ( *v29 != (_DWORD)result )
  {
    v39 = 1;
    v35 = 0;
    while ( v30 != -1 )
    {
      if ( v30 != -2 || v35 )
        v29 = v35;
      v28 = v27 & (v39 + v28);
      v30 = *(_DWORD *)(v26 + 24LL * v28);
      if ( (_DWORD)result == v30 )
        goto LABEL_31;
      ++v39;
      v35 = v29;
      v29 = (_DWORD *)(v26 + 24LL * v28);
    }
    v34 = *(_DWORD *)(a1 + 352);
    if ( !v35 )
      v35 = v29;
    ++*(_QWORD *)(a1 + 344);
    v36 = (v34 >> 1) + 1;
    if ( (_BYTE)v25 )
    {
      v37 = 24;
      v31 = 8;
LABEL_42:
      if ( 4 * v36 < v37 )
      {
        if ( v31 - *(_DWORD *)(a1 + 356) - v36 > v31 >> 3 )
        {
LABEL_44:
          *(_DWORD *)(a1 + 352) = (2 * (v34 >> 1) + 2) | v34 & 1;
          if ( *v35 != -1 )
            --*(_DWORD *)(a1 + 356);
          v77.m128i_i64[0] = a2;
          v77.m128i_i32[2] = a3;
          v38 = _mm_loadu_si128(&v77);
          *v35 = result;
          *(__m128i *)(v35 + 2) = v38;
          result = *(unsigned int *)(a1 + 132);
          goto LABEL_31;
        }
        v72 = a3;
        v76 = result;
        sub_200F500(a1 + 344, v31);
        LODWORD(result) = v76;
        LODWORD(a3) = v72;
        if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
        {
          v55 = a1 + 360;
          v56 = 7;
          goto LABEL_73;
        }
        v65 = *(_DWORD *)(a1 + 368);
        v55 = *(_QWORD *)(a1 + 360);
        if ( v65 )
        {
          v56 = v65 - 1;
LABEL_73:
          v57 = v56 & (37 * v76);
          v35 = (_DWORD *)(v55 + 24LL * v57);
          v58 = *v35;
          if ( *v35 != v76 )
          {
            v59 = 1;
            v60 = 0;
            while ( v58 != -1 )
            {
              if ( v58 == -2 && !v60 )
                v60 = v35;
              v57 = v56 & (v59 + v57);
              v35 = (_DWORD *)(v55 + 24LL * v57);
              v58 = *v35;
              if ( v76 == *v35 )
                goto LABEL_64;
              ++v59;
            }
LABEL_76:
            if ( v60 )
              v35 = v60;
            goto LABEL_64;
          }
          goto LABEL_64;
        }
LABEL_129:
        *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
        BUG();
      }
      v70 = a3;
      v74 = result;
      sub_200F500(a1 + 344, 2 * v31);
      LODWORD(result) = v74;
      LODWORD(a3) = v70;
      if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
      {
        v46 = a1 + 360;
        v47 = 7;
      }
      else
      {
        v62 = *(_DWORD *)(a1 + 368);
        v46 = *(_QWORD *)(a1 + 360);
        if ( !v62 )
          goto LABEL_129;
        v47 = v62 - 1;
      }
      v48 = v47 & (37 * v74);
      v35 = (_DWORD *)(v46 + 24LL * v48);
      v49 = *v35;
      if ( v74 != *v35 )
      {
        v66 = 1;
        v60 = 0;
        while ( v49 != -1 )
        {
          if ( v49 == -2 && !v60 )
            v60 = v35;
          v48 = v47 & (v66 + v48);
          v35 = (_DWORD *)(v46 + 24LL * v48);
          v49 = *v35;
          if ( v74 == *v35 )
            goto LABEL_64;
          ++v66;
        }
        goto LABEL_76;
      }
LABEL_64:
      v34 = *(_DWORD *)(a1 + 352);
      goto LABEL_44;
    }
    v31 = *(_DWORD *)(a1 + 368);
LABEL_41:
    v37 = 3 * v31;
    goto LABEL_42;
  }
LABEL_31:
  *(_DWORD *)(a1 + 132) = result + 1;
  return result;
}
