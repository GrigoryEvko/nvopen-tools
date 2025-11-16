// Function: sub_B25B40
// Address: 0xb25b40
//
void __fastcall sub_B25B40(unsigned __int64 *a1, __int64 a2, __int64 a3, char a4, char a5)
{
  _BYTE *v5; // rax
  char v7; // r13
  unsigned int v8; // eax
  unsigned __int64 *v9; // rbx
  unsigned int v10; // esi
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  int v15; // r12d
  bool v16; // zf
  __int64 *v17; // rax
  unsigned int v18; // edx
  unsigned int v19; // eax
  unsigned int v20; // eax
  char v21; // dl
  __int64 v22; // rax
  _QWORD *v23; // r8
  _QWORD *v24; // r13
  _BYTE *v25; // r9
  _QWORD *v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // r14
  _QWORD *v29; // r15
  int v30; // eax
  __int64 v31; // r8
  unsigned __int64 v32; // r12
  __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rbx
  __int64 *v36; // r12
  unsigned __int64 v37; // rax
  __m128i *v38; // r12
  __int64 v39; // rbx
  unsigned __int64 v40; // rax
  const __m128i *v41; // r13
  __m128i *v42; // r14
  __m128i v43; // xmm0
  _QWORD *v44; // r13
  __int64 v45; // rax
  __int64 v46; // [rsp+10h] [rbp-100h]
  __int64 v47; // [rsp+10h] [rbp-100h]
  unsigned __int64 *v48; // [rsp+20h] [rbp-F0h]
  __int64 v49; // [rsp+20h] [rbp-F0h]
  __int64 v51; // [rsp+30h] [rbp-E0h]
  __m128i *v52; // [rsp+30h] [rbp-E0h]
  _BYTE v53[12]; // [rsp+3Ch] [rbp-D4h] BYREF
  __int64 *v54; // [rsp+48h] [rbp-C8h] BYREF
  __int64 *v55[2]; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v56; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v57; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v58; // [rsp+78h] [rbp-98h]
  _QWORD *v59; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v60; // [rsp+88h] [rbp-88h]
  _BYTE v61[48]; // [rsp+E0h] [rbp-30h] BYREF

  v5 = &v59;
  v51 = a2;
  v53[0] = a5;
  v57 = 0;
  v58 = 1;
  do
  {
    *(_QWORD *)v5 = -4096;
    v5 += 24;
    *((_QWORD *)v5 - 2) = -4096;
  }
  while ( v5 != v61 );
  v7 = v58 & 1;
  if ( (_DWORD)a2 )
  {
    ++v57;
    LODWORD(a2) = sub_AF1560(4 * (int)a2 / 3u + 1);
    v8 = 4;
    if ( v7 )
      goto LABEL_6;
    goto LABEL_5;
  }
  ++v57;
  if ( !v7 )
  {
LABEL_5:
    v8 = v60;
LABEL_6:
    if ( v8 < (unsigned int)a2 )
      sub_B1DB20((__int64)&v57, a2);
  }
  v48 = &a1[2 * v51];
  if ( a1 == v48 )
    goto LABEL_26;
  v46 = a3;
  v9 = a1;
  do
  {
    v12 = v9[1];
    v13 = *v9;
    v14 = v12 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !a4 )
    {
      v14 = *v9;
      v13 = v9[1] & 0xFFFFFFFFFFFFFFF8LL;
    }
    v56.m128i_i64[0] = v14;
    v56.m128i_i64[1] = v13;
    v15 = (v12 & 4) == 0 ? 1 : -1;
    v16 = (unsigned __int8)sub_B1C410((__int64)&v57, v56.m128i_i64, &v54) == 0;
    v17 = v54;
    if ( v16 )
    {
      ++v57;
      v55[0] = v54;
      v18 = ((unsigned int)v58 >> 1) + 1;
      if ( (v58 & 1) != 0 )
      {
        v10 = 4;
        if ( 4 * v18 >= 0xC )
          goto LABEL_21;
      }
      else
      {
        v10 = v60;
        if ( 4 * v18 >= 3 * v60 )
        {
LABEL_21:
          v10 *= 2;
LABEL_22:
          sub_B1DB20((__int64)&v57, v10);
          sub_B1C410((__int64)&v57, v56.m128i_i64, v55);
          v17 = v55[0];
          v18 = ((unsigned int)v58 >> 1) + 1;
LABEL_12:
          LODWORD(v58) = v58 & 1 | (2 * v18);
          if ( *v17 != -4096 || v17[1] != -4096 )
            --HIDWORD(v58);
          *v17 = v56.m128i_i64[0];
          v11 = v56.m128i_i64[1];
          *((_DWORD *)v17 + 4) = 0;
          v17[1] = v11;
          goto LABEL_15;
        }
      }
      if ( v10 - (v18 + HIDWORD(v58)) > v10 >> 3 )
        goto LABEL_12;
      goto LABEL_22;
    }
LABEL_15:
    v9 += 2;
    *((_DWORD *)v17 + 4) += v15;
  }
  while ( v48 != v9 );
  a3 = v46;
LABEL_26:
  v19 = v58;
  *(_DWORD *)(a3 + 8) = 0;
  v20 = v19 >> 1;
  if ( *(_DWORD *)(a3 + 12) < v20 )
  {
    sub_C8D5F0(a3, a3 + 16, v20, 16);
    v20 = (unsigned int)v58 >> 1;
  }
  v21 = v58 & 1;
  if ( v20 )
  {
    if ( v21 )
    {
      v26 = v59;
      v24 = &v59;
      v25 = v61;
      if ( v59 != (_QWORD *)-4096LL )
        goto LABEL_32;
      goto LABEL_55;
    }
    v22 = v60;
    v23 = v59;
    v24 = v59;
    v25 = &v59[3 * v60];
    if ( v25 != (_BYTE *)v59 )
    {
      while ( 1 )
      {
        v26 = (_QWORD *)*v24;
        if ( *v24 == -4096 )
        {
LABEL_55:
          if ( v24[1] != -4096 )
            goto LABEL_33;
        }
        else
        {
LABEL_32:
          if ( v26 != (_QWORD *)-8192LL || v24[1] != -8192 )
            goto LABEL_33;
        }
        v24 += 3;
        if ( v24 == (_QWORD *)v25 )
          goto LABEL_33;
      }
    }
LABEL_35:
    v27 = 3 * v22;
  }
  else
  {
    if ( v21 )
    {
      v44 = &v59;
      v45 = 12;
    }
    else
    {
      v44 = v59;
      v45 = 3LL * v60;
    }
    v24 = &v44[v45];
    v25 = v24;
LABEL_33:
    if ( !v21 )
    {
      v23 = v59;
      v22 = v60;
      goto LABEL_35;
    }
    v23 = &v59;
    v27 = 12;
  }
  v28 = &v23[v27];
  v29 = v25;
  while ( v24 != v28 )
  {
LABEL_40:
    v30 = *((_DWORD *)v24 + 4);
    if ( v30 )
    {
      v31 = *v24;
      v32 = (4LL * (v30 <= 0)) | v24[1] & 0xFFFFFFFFFFFFFFFBLL;
      v33 = *(unsigned int *)(a3 + 8);
      if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v47 = *v24;
        sub_C8D5F0(a3, a3 + 16, v33 + 1, 16);
        v33 = *(unsigned int *)(a3 + 8);
        v31 = v47;
      }
      v34 = (_QWORD *)(*(_QWORD *)a3 + 16 * v33);
      *v34 = v31;
      v34[1] = v32;
      ++*(_DWORD *)(a3 + 8);
    }
    do
    {
      while ( 1 )
      {
        v24 += 3;
        if ( v24 == v29 )
        {
LABEL_39:
          if ( v24 == v28 )
            goto LABEL_48;
          goto LABEL_40;
        }
        if ( *v24 == -4096 )
          break;
        if ( *v24 != -8192 || v24[1] != -8192 )
          goto LABEL_39;
      }
    }
    while ( v24[1] == -4096 );
  }
LABEL_48:
  if ( v51 )
  {
    v49 = a3;
    v35 = 0;
    v36 = (__int64 *)a1;
    do
    {
      v37 = v36[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( a4 )
      {
        v56.m128i_i64[1] = *v36;
        v56.m128i_i64[0] = v37;
      }
      else
      {
        v56.m128i_i64[0] = *v36;
        v56.m128i_i64[1] = v37;
      }
      *(_DWORD *)sub_B1DDD0((__int64)&v57, v56.m128i_i64) = v35++;
      v36 += 2;
    }
    while ( v35 != v51 );
    a3 = v49;
  }
  v38 = *(__m128i **)a3;
  v39 = *(unsigned int *)(a3 + 8);
  v52 = &v38[v39];
  if ( v38 != &v38[v39] )
  {
    _BitScanReverse64(&v40, (v39 * 16) >> 4);
    sub_B25040(v38, v52->m128i_i64, 2LL * (int)(63 - (v40 ^ 0x3F)), (__int64)&v57, v53, (__int64)v25);
    if ( (unsigned __int64)v39 <= 16 )
    {
      sub_B23600(v38, v52, (__int64)&v57, (__int64)v53);
    }
    else
    {
      sub_B23600(v38, v38 + 16, (__int64)&v57, (__int64)v53);
      if ( v52 != &v38[16] )
      {
        v41 = v38 + 16;
        do
        {
          v42 = (__m128i *)&v41[-1];
          v55[0] = &v57;
          v55[1] = (__int64 *)v53;
          v56 = _mm_loadu_si128(v41);
          while ( sub_B1DED0((__int64)v55, v56.m128i_i64, v42->m128i_i64) )
          {
            v43 = _mm_loadu_si128(v42--);
            v42[2] = v43;
          }
          ++v41;
          v42[1] = _mm_load_si128(&v56);
        }
        while ( v52 != v41 );
      }
    }
  }
  if ( (v58 & 1) == 0 )
    sub_C7D6A0(v59, 24LL * v60, 8);
}
