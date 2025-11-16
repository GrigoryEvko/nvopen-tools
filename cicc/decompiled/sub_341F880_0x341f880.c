// Function: sub_341F880
// Address: 0x341f880
//
__int64 __fastcall sub_341F880(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, _QWORD **a5)
{
  _QWORD *v6; // rsi
  bool v9; // r8
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // r10
  unsigned __int8 *v13; // rsi
  unsigned __int8 v14; // al
  unsigned int v15; // r8d
  int v17; // r12d
  unsigned int v18; // r14d
  _QWORD *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  unsigned int v29; // r10d
  __int64 *v30; // r9
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r13
  __int64 v37; // r12
  __int64 v38; // rax
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rsi
  __int64 v41; // rcx
  int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rcx
  unsigned int v46; // r10d
  __int64 v47; // r9
  unsigned __int8 *v48; // rdi
  unsigned __int64 v49; // rdx
  const __m128i *v50; // rcx
  __int64 v51; // rdx
  __m128i *v52; // rax
  int v53; // edx
  int v54; // r11d
  __int64 v55; // rdx
  unsigned __int64 v56; // rdx
  const __m128i *v57; // rax
  __int64 v58; // rdx
  __m128i *v59; // rdx
  int v60; // edx
  int v61; // r11d
  __int64 v62; // rdx
  unsigned __int64 v63; // rbx
  __int64 v64; // rdi
  const void *v65; // rsi
  unsigned __int64 v66; // r12
  __int64 v67; // r9
  const void *v68; // rsi
  __int64 v69; // [rsp+8h] [rbp-78h]
  bool v70; // [rsp+8h] [rbp-78h]
  __int64 v71; // [rsp+10h] [rbp-70h]
  int v72; // [rsp+10h] [rbp-70h]
  unsigned __int8 v74; // [rsp+18h] [rbp-68h]
  unsigned __int8 v75; // [rsp+18h] [rbp-68h]
  _QWORD *v76; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v77; // [rsp+28h] [rbp-58h]
  __int64 v78; // [rsp+30h] [rbp-50h] BYREF
  __int64 v79; // [rsp+38h] [rbp-48h]
  _QWORD *v80; // [rsp+40h] [rbp-40h]
  __int64 v81; // [rsp+48h] [rbp-38h]

  if ( !a2 )
    return 0;
  v6 = *a5;
  v76 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v76, (__int64)v6, 1);
  v9 = sub_AF46F0((__int64)a3);
  if ( !v9 || *(_BYTE *)a2 != 22 || (v27 = *(unsigned int *)(a1 + 144), v28 = *(_QWORD *)(a1 + 128), !(_DWORD)v27) )
  {
LABEL_6:
    if ( v76 )
      sub_B91220((__int64)&v76, (__int64)v76);
    v71 = *(_QWORD *)(a1 + 8);
    v69 = sub_2E79000((__int64 *)v71);
    v78 = sub_9208B0(v69, *(_QWORD *)(a2 + 8));
    v79 = v10;
    v11 = sub_CA1930(&v78);
    v12 = v69;
    v77 = v11;
    if ( v11 > 0x40 )
    {
      sub_C43690((__int64)&v76, 0, 0);
      v12 = v69;
    }
    else
    {
      v76 = 0;
    }
    v13 = sub_BD45C0((unsigned __int8 *)a2, v12, (__int64)&v76, 0, 0, 0, 0, 0);
    v14 = *v13;
    if ( *v13 <= 0x1Cu )
    {
      if ( v14 != 22 )
        goto LABEL_12;
      v17 = sub_374D370(a1, v13);
    }
    else
    {
      if ( v14 != 60 )
        goto LABEL_12;
      v44 = *(unsigned int *)(a1 + 272);
      v45 = *(_QWORD *)(a1 + 256);
      if ( !(_DWORD)v44 )
        goto LABEL_12;
      v46 = (v44 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v47 = v45 + 16LL * v46;
      v48 = *(unsigned __int8 **)v47;
      if ( *(unsigned __int8 **)v47 != v13 )
      {
        v60 = 1;
        while ( v48 != (unsigned __int8 *)-4096LL )
        {
          v61 = v60 + 1;
          v62 = ((_DWORD)v44 - 1) & (v46 + v60);
          v46 = v62;
          v47 = v45 + 16 * v62;
          v48 = *(unsigned __int8 **)v47;
          if ( v13 == *(unsigned __int8 **)v47 )
            goto LABEL_43;
          v60 = v61;
        }
        goto LABEL_12;
      }
LABEL_43:
      if ( v47 == v45 + 16 * v44 )
      {
LABEL_12:
        v15 = 0;
LABEL_13:
        if ( v77 > 0x40 && v76 )
        {
          v74 = v15;
          j_j___libc_free_0_0((unsigned __int64)v76);
          return v74;
        }
        return v15;
      }
      v17 = *(_DWORD *)(v47 + 8);
    }
    if ( v17 != 0x7FFFFFFF )
    {
      v18 = v77;
      if ( v77 <= 0x40 )
      {
        v19 = v76;
        if ( !v76 )
        {
LABEL_24:
          v20 = sub_B10CD0((__int64)a5);
          v23 = *(unsigned int *)(v71 + 764);
          v24 = *(unsigned int *)(v71 + 760);
          v25 = *(_DWORD *)(v71 + 760);
          if ( v24 >= v23 )
          {
            v49 = v24 + 1;
            LODWORD(v78) = v17;
            BYTE4(v78) = 0;
            v79 = a4;
            v80 = a3;
            v81 = v20;
            if ( v23 < v24 + 1 )
            {
              v63 = *(_QWORD *)(v71 + 752);
              v64 = v71 + 752;
              v65 = (const void *)(v71 + 768);
              if ( v63 > (unsigned __int64)&v78 || (unsigned __int64)&v78 >= v63 + 32 * v24 )
              {
                sub_C8D5F0(v64, v65, v49, 0x20u, v21, v22);
                v50 = (const __m128i *)&v78;
                v24 = *(unsigned int *)(v71 + 760);
                v51 = *(_QWORD *)(v71 + 752);
              }
              else
              {
                sub_C8D5F0(v64, v65, v49, 0x20u, v21, v22);
                v51 = *(_QWORD *)(v71 + 752);
                v24 = *(unsigned int *)(v71 + 760);
                v50 = (const __m128i *)((char *)&v78 + v51 - v63);
              }
            }
            else
            {
              v50 = (const __m128i *)&v78;
              v51 = *(_QWORD *)(v71 + 752);
            }
            v15 = 1;
            v52 = (__m128i *)(v51 + 32 * v24);
            *v52 = _mm_loadu_si128(v50);
            v52[1] = _mm_loadu_si128(v50 + 1);
            ++*(_DWORD *)(v71 + 760);
          }
          else
          {
            v26 = *(_QWORD *)(v71 + 752) + 32 * v24;
            if ( v26 )
            {
              *(_DWORD *)v26 = v17;
              *(_BYTE *)(v26 + 4) = 0;
              *(_QWORD *)(v26 + 8) = a4;
              *(_QWORD *)(v26 + 16) = a3;
              *(_QWORD *)(v26 + 24) = v20;
              v25 = *(_DWORD *)(v71 + 760);
            }
            v15 = 1;
            *(_DWORD *)(v71 + 760) = v25 + 1;
          }
          goto LABEL_13;
        }
      }
      else
      {
        if ( v18 == (unsigned int)sub_C444A0((__int64)&v76) )
          goto LABEL_24;
        v19 = (_QWORD *)*v76;
      }
      a3 = (_QWORD *)sub_B0DAC0(a3, 0, (__int64)v19);
      goto LABEL_24;
    }
    goto LABEL_12;
  }
  v29 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v30 = (__int64 *)(v28 + 16LL * v29);
  v31 = *v30;
  if ( a2 != *v30 )
  {
    v53 = 1;
    while ( v31 != -4096 )
    {
      v54 = v53 + 1;
      v55 = ((_DWORD)v27 - 1) & (v29 + v53);
      v29 = v55;
      v30 = (__int64 *)(v28 + 16 * v55);
      v31 = *v30;
      if ( a2 == *v30 )
        goto LABEL_30;
      v53 = v54;
    }
    goto LABEL_6;
  }
LABEL_30:
  if ( v30 == (__int64 *)(v28 + 16 * v27) )
    goto LABEL_6;
  v32 = *(_QWORD *)(a1 + 24);
  v33 = *(_QWORD *)(v32 + 496);
  v34 = *(_QWORD *)(v32 + 488);
  if ( v34 == v33 )
    goto LABEL_6;
  while ( *((_DWORD *)v30 + 2) != *(_DWORD *)(v34 + 4) )
  {
    v34 += 8;
    if ( v33 == v34 )
      goto LABEL_6;
  }
  v70 = v9;
  v72 = *(_QWORD *)v34;
  v78 = 6;
  v35 = sub_B0DED0(a3, &v78, 1);
  v36 = *(_QWORD *)(a1 + 8);
  v37 = v35;
  v38 = sub_B10CD0((__int64)&v76);
  v39 = *(unsigned int *)(v36 + 760);
  v40 = *(unsigned int *)(v36 + 764);
  v15 = v70;
  v41 = v38;
  v42 = *(_DWORD *)(v36 + 760);
  if ( v39 >= v40 )
  {
    v56 = v39 + 1;
    LODWORD(v78) = v72;
    BYTE4(v78) = 1;
    v79 = a4;
    v80 = (_QWORD *)v37;
    v81 = v38;
    if ( v40 < v39 + 1 )
    {
      v66 = *(_QWORD *)(v36 + 752);
      v67 = v36 + 752;
      v68 = (const void *)(v36 + 768);
      if ( v66 > (unsigned __int64)&v78 || (unsigned __int64)&v78 >= v66 + 32 * v39 )
      {
        sub_C8D5F0(v36 + 752, v68, v56, 0x20u, v70, v67);
        v39 = *(unsigned int *)(v36 + 760);
        v57 = (const __m128i *)&v78;
        v58 = *(_QWORD *)(v36 + 752);
        v15 = v70;
      }
      else
      {
        sub_C8D5F0(v36 + 752, v68, v56, 0x20u, v70, v67);
        v58 = *(_QWORD *)(v36 + 752);
        v39 = *(unsigned int *)(v36 + 760);
        v15 = v70;
        v57 = (const __m128i *)((char *)&v78 + v58 - v66);
      }
    }
    else
    {
      v57 = (const __m128i *)&v78;
      v58 = *(_QWORD *)(v36 + 752);
    }
    v59 = (__m128i *)(32 * v39 + v58);
    *v59 = _mm_loadu_si128(v57);
    v59[1] = _mm_loadu_si128(v57 + 1);
    ++*(_DWORD *)(v36 + 760);
  }
  else
  {
    v43 = *(_QWORD *)(v36 + 752) + 32 * v39;
    if ( v43 )
    {
      *(_DWORD *)v43 = v72;
      *(_BYTE *)(v43 + 4) = 1;
      *(_QWORD *)(v43 + 8) = a4;
      *(_QWORD *)(v43 + 16) = v37;
      *(_QWORD *)(v43 + 24) = v41;
      v42 = *(_DWORD *)(v36 + 760);
    }
    *(_DWORD *)(v36 + 760) = v42 + 1;
  }
  if ( v76 )
  {
    v75 = v15;
    sub_B91220((__int64)&v76, (__int64)v76);
    return v75;
  }
  return v15;
}
