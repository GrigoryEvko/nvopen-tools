// Function: sub_37C6BA0
// Address: 0x37c6ba0
//
__int64 __fastcall sub_37C6BA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  _QWORD *v8; // rcx
  __int64 v10; // r12
  __int64 v11; // r8
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // rax
  unsigned int v15; // r13d
  __int64 v16; // rdx
  char v17; // al
  int v19; // r14d
  __int64 v20; // r9
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // r12
  __m128i *v24; // r9
  __int64 v25; // rax
  __int64 m128i_i64; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r9
  __int64 v30; // r12
  _QWORD *v31; // rcx
  char v32; // al
  char v33; // di
  __m128i *v34; // rsi
  unsigned int v35; // r12d
  unsigned int v36; // eax
  char *v37; // rsi
  __int64 v38; // rcx
  int v39; // eax
  int v40; // eax
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // r11
  unsigned int v43; // ecx
  __int64 v44; // r8
  unsigned __int64 v45; // rsi
  __int64 v46; // r9
  __int64 *v47; // rdx
  __int64 v48; // rcx
  int v49; // edi
  __int64 v50; // r8
  __int64 v51; // r9
  _DWORD *v52; // rdx
  _DWORD *v53; // rcx
  __int64 v54; // [rsp+0h] [rbp-70h]
  __int64 v55; // [rsp+0h] [rbp-70h]
  int v56; // [rsp+8h] [rbp-68h]
  __int64 v57; // [rsp+8h] [rbp-68h]
  unsigned __int64 v58; // [rsp+18h] [rbp-58h]
  __int64 v59; // [rsp+20h] [rbp-50h]
  _QWORD *v60; // [rsp+20h] [rbp-50h]
  _QWORD *v61; // [rsp+20h] [rbp-50h]
  __int64 v62; // [rsp+28h] [rbp-48h]
  unsigned __int64 v63; // [rsp+28h] [rbp-48h]
  __int64 v64; // [rsp+28h] [rbp-48h]
  __int64 v65; // [rsp+28h] [rbp-48h]
  int v66; // [rsp+28h] [rbp-48h]
  int v67; // [rsp+28h] [rbp-48h]
  int v68; // [rsp+28h] [rbp-48h]
  __int64 v69; // [rsp+28h] [rbp-48h]
  __int64 v70; // [rsp+38h] [rbp-38h] BYREF

  v8 = (_QWORD *)(a1 + 216);
  v10 = *(_QWORD *)(a1 + 224);
  if ( !v10 )
  {
    v20 = a1 + 216;
    if ( (unsigned int)qword_5051208 <= 0xAAAAAAAAAAAAAAABLL
                                      * ((__int64)(*(_QWORD *)(a1 + 264) - *(_QWORD *)(a1 + 256)) >> 3) )
      goto LABEL_13;
LABEL_28:
    v23 = v20;
    v61 = v8;
    v24 = (__m128i *)sub_22077B0(0x40u);
    v24[2] = _mm_loadu_si128((const __m128i *)&a7);
    v25 = a8;
    m128i_i64 = (__int64)v24[2].m128i_i64;
    v24[3].m128i_i32[2] = 0;
    v24[3].m128i_i64[0] = v25;
    v63 = (unsigned __int64)v24;
    v27 = sub_37C6A40((_QWORD *)(a1 + 208), v23, (__int64)v24[2].m128i_i64);
    v29 = v63;
    v30 = v27;
    if ( v28 )
    {
      v31 = v61;
      if ( v61 == (_QWORD *)v28 || v27 )
      {
        v33 = 1;
      }
      else
      {
        v58 = v63;
        v64 = v28;
        v32 = sub_37B9A70(m128i_i64, (int *)(v28 + 32));
        v28 = v64;
        v31 = v61;
        v29 = v58;
        v33 = v32;
      }
      v65 = v29;
      sub_220F040(v33, v29, (_QWORD *)v28, v31);
      ++*(_QWORD *)(a1 + 248);
      v20 = v65;
    }
    else
    {
      j_j___libc_free_0(v63);
      v20 = v30;
    }
LABEL_35:
    v19 = *(_DWORD *)(v20 + 56);
    if ( !v19 )
    {
      *(_DWORD *)(v20 + 56) = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 264) - *(_QWORD *)(a1 + 256)) >> 3) + 1;
      v34 = *(__m128i **)(a1 + 264);
      if ( v34 == *(__m128i **)(a1 + 272) )
      {
        v69 = v20;
        sub_37BF670((unsigned __int64 *)(a1 + 256), v34, (const __m128i *)&a7);
        v20 = v69;
      }
      else
      {
        if ( v34 )
        {
          *v34 = _mm_loadu_si128((const __m128i *)&a7);
          v34[1].m128i_i64[0] = a8;
          v34 = *(__m128i **)(a1 + 264);
        }
        *(_QWORD *)(a1 + 264) = (char *)v34 + 24;
      }
      v19 = *(_DWORD *)(v20 + 56);
    }
    if ( !*(_DWORD *)(a1 + 288) )
      goto LABEL_17;
    v35 = 0;
    v36 = *(_DWORD *)(a1 + 288);
    while ( 1 )
    {
      v40 = v35 + *(_DWORD *)(a1 + 284) + (v19 - 1) * v36;
      v41 = *(unsigned int *)(a1 + 40);
      v42 = (unsigned int)(v41 + 1);
      LODWORD(v70) = *(_DWORD *)(a1 + 40);
      v43 = v42;
      if ( (unsigned int)v41 >= (unsigned int)v42 )
        goto LABEL_43;
      v44 = *(_QWORD *)(a1 + 48);
      if ( v42 != v41 )
      {
        if ( v42 >= v41 )
        {
          v46 = v42 - v41;
          if ( v42 > *(unsigned int *)(a1 + 44) )
          {
            v55 = *(_QWORD *)(a1 + 48);
            v57 = v42 - v41;
            v68 = v40;
            sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), (unsigned int)(v41 + 1), 8u, v44, v46);
            v41 = *(unsigned int *)(a1 + 40);
            v44 = v55;
            v46 = v57;
            v40 = v68;
          }
          v47 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8 * v41);
          v48 = v46;
          do
          {
            if ( v47 )
              *v47 = v44;
            ++v47;
            --v48;
          }
          while ( v48 );
          v49 = v70;
          *(_DWORD *)(a1 + 40) += v46;
          v41 = (unsigned int)(v49 + 1);
          v43 = v49 + 1;
        }
        else
        {
          *(_DWORD *)(a1 + 40) = v42;
          v41 = (unsigned int)(v41 + 1);
        }
      }
      v45 = *(unsigned int *)(a1 + 96);
      if ( (unsigned int)v45 >= v43 || v41 == v45 )
        goto LABEL_43;
      if ( v41 >= v45 )
        break;
      *(_DWORD *)(a1 + 96) = v43;
      v37 = *(char **)(a1 + 72);
      if ( v37 != *(char **)(a1 + 80) )
      {
LABEL_44:
        if ( v37 )
        {
          *(_DWORD *)v37 = v70;
          v37 = *(char **)(a1 + 72);
        }
        *(_QWORD *)(a1 + 72) = v37 + 4;
        goto LABEL_47;
      }
LABEL_56:
      v66 = v40;
      sub_37BDD10((unsigned __int64 *)(a1 + 64), v37, &v70);
      v40 = v66;
LABEL_47:
      ++v35;
      *(_DWORD *)(*(_QWORD *)(a1 + 88) + 4LL * (unsigned int)v70) = v40;
      v38 = *(_QWORD *)(a1 + 32) + 8LL * (unsigned int)v70;
      v39 = (_DWORD)v70 << 8;
      *(_QWORD *)v38 = *(_QWORD *)v38 & 0xFFFFFF0000000000LL | *(_DWORD *)(a1 + 280) & 0xFFFFF;
      *(_DWORD *)(v38 + 4) = v39;
      v36 = *(_DWORD *)(a1 + 288);
      if ( v36 <= v35 )
        goto LABEL_17;
    }
    v50 = *(unsigned int *)(a1 + 104);
    v51 = v41 - v45;
    if ( v41 > *(unsigned int *)(a1 + 100) )
    {
      v67 = v40;
      v54 = v41 - v45;
      v56 = *(_DWORD *)(a1 + 104);
      sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), v41, 4u, v50, v51);
      v45 = *(unsigned int *)(a1 + 96);
      v51 = v54;
      LODWORD(v50) = v56;
      v40 = v67;
    }
    v52 = (_DWORD *)(*(_QWORD *)(a1 + 88) + 4 * v45);
    v53 = &v52[v51];
    if ( v52 != v53 )
    {
      do
        *v52++ = v50;
      while ( v53 != v52 );
      LODWORD(v45) = *(_DWORD *)(a1 + 96);
    }
    *(_DWORD *)(a1 + 96) = v51 + v45;
LABEL_43:
    v37 = *(char **)(a1 + 72);
    if ( v37 != *(char **)(a1 + 80) )
      goto LABEL_44;
    goto LABEL_56;
  }
  v11 = a8;
  v12 = *((_QWORD *)&a7 + 1);
  v13 = a1 + 216;
  v14 = *(_QWORD *)(a1 + 224);
  v15 = a7;
  do
  {
    if ( *(_DWORD *)(v14 + 32) < (unsigned int)a7
      || (v16 = *(_QWORD *)(v14 + 40), *(_DWORD *)(v14 + 32) == (_DWORD)a7)
      && (v16 < *((__int64 *)&a7 + 1) || v16 == *((_QWORD *)&a7 + 1) && *(_QWORD *)(v14 + 48) < a8) )
    {
      v14 = *(_QWORD *)(v14 + 24);
    }
    else
    {
      v13 = v14;
      v14 = *(_QWORD *)(v14 + 16);
    }
  }
  while ( v14 );
  if ( v8 != (_QWORD *)v13 )
  {
    v59 = a8;
    v17 = sub_37B9A70((__int64)&a7, (int *)(v13 + 32));
    v8 = (_QWORD *)(a1 + 216);
    v11 = v59;
    if ( !v17 )
    {
      v19 = *(_DWORD *)(v13 + 56);
      if ( v19 )
      {
LABEL_17:
        LODWORD(v70) = v19;
        BYTE4(v70) = 1;
        return v70;
      }
    }
  }
  if ( 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 264) - *(_QWORD *)(a1 + 256)) >> 3) < (unsigned int)qword_5051208 )
  {
    v20 = (__int64)v8;
    do
    {
      if ( *(_DWORD *)(v10 + 32) < v15
        || (v21 = *(_QWORD *)(v10 + 40), *(_DWORD *)(v10 + 32) == v15)
        && (v12 > v21 || v12 == v21 && *(_QWORD *)(v10 + 48) < v11) )
      {
        v10 = *(_QWORD *)(v10 + 24);
      }
      else
      {
        v20 = v10;
        v10 = *(_QWORD *)(v10 + 16);
      }
    }
    while ( v10 );
    if ( v8 != (_QWORD *)v20 )
    {
      v60 = v8;
      v62 = v20;
      v22 = sub_37B9A70((__int64)&a7, (int *)(v20 + 32));
      v20 = v62;
      v8 = v60;
      if ( !v22 )
        goto LABEL_35;
    }
    goto LABEL_28;
  }
LABEL_13:
  BYTE4(v70) = 0;
  return v70;
}
