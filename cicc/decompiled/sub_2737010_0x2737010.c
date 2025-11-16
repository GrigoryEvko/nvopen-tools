// Function: sub_2737010
// Address: 0x2737010
//
__int64 __fastcall sub_2737010(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned int v7; // r12d
  __int64 v8; // r9
  __int64 v9; // rdx
  unsigned int v10; // r14d
  __int64 v11; // r13
  __int64 *v12; // r12
  __int64 v13; // rbx
  _BYTE *v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned __int64 v17; // rdi
  char v18; // r11
  int v19; // ecx
  __int64 v20; // r15
  char v21; // bl
  __int64 v22; // r10
  __int64 v23; // r9
  unsigned __int64 v24; // rax
  _BYTE *v25; // rdi
  unsigned __int64 v27; // rdx
  const __m128i *v28; // rbx
  __m128i *v29; // rdi
  __m128i v30; // xmm1
  __int64 v31; // rax
  __int64 v32; // r12
  unsigned __int16 v33; // bx
  __int64 v34; // r14
  _QWORD *v35; // rax
  __int64 v36; // r13
  __int64 v37; // rsi
  __int64 *v38; // r12
  __int64 v39; // rbx
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  __int64 v42; // r15
  __int64 v43; // rax
  _QWORD *v44; // rax
  char *v45; // rbx
  __int64 v46; // rsi
  unsigned __int8 *v47; // rsi
  __int64 v48; // r12
  unsigned __int16 v49; // bx
  __int64 v50; // r14
  _QWORD *v51; // rax
  __int64 v52; // [rsp+18h] [rbp-218h]
  __int64 *v53; // [rsp+30h] [rbp-200h]
  __int64 *v54; // [rsp+38h] [rbp-1F8h]
  __int64 v55; // [rsp+50h] [rbp-1E0h]
  __int64 *v57; // [rsp+60h] [rbp-1D0h]
  __int64 v58; // [rsp+68h] [rbp-1C8h]
  __int64 v59; // [rsp+70h] [rbp-1C0h]
  _BYTE *v60; // [rsp+70h] [rbp-1C0h]
  __int64 v61; // [rsp+78h] [rbp-1B8h] BYREF
  char v62[8]; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v63; // [rsp+88h] [rbp-1A8h]
  unsigned int v64; // [rsp+98h] [rbp-198h]
  __int64 *v65; // [rsp+A0h] [rbp-190h]
  unsigned int v66; // [rsp+A8h] [rbp-188h]
  __int64 v67[3]; // [rsp+B0h] [rbp-180h] BYREF
  char v68; // [rsp+C8h] [rbp-168h]
  char v69; // [rsp+C9h] [rbp-167h]
  __int64 v70; // [rsp+D0h] [rbp-160h]
  int v71; // [rsp+D8h] [rbp-158h]
  _BYTE *v72; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v73; // [rsp+E8h] [rbp-148h]
  _BYTE v74[64]; // [rsp+F0h] [rbp-140h] BYREF
  unsigned __int64 v75; // [rsp+130h] [rbp-100h] BYREF
  __int64 v76; // [rsp+138h] [rbp-F8h]
  _BYTE v77[240]; // [rsp+140h] [rbp-F0h] BYREF

  v6 = (__int64)(a1 + 17);
  v61 = a2;
  if ( a2 )
    v6 = sub_2735710((__int64)(a1 + 691), &v61, a3, a4, a5, a6);
  v7 = 0;
  v54 = *(__int64 **)v6;
  v52 = *(_QWORD *)v6 + 672LL * *(unsigned int *)(v6 + 8);
  if ( *(_QWORD *)v6 != v52 )
  {
    while ( 1 )
    {
      v72 = v74;
      v73 = 0x400000000LL;
      sub_2730890((__int64)a1, (__int64)(v54 + 2), (__int64)&v72);
      sub_2733E60((__int64)v62, a1, (__int64)v54, (__int64)v72, (unsigned int)v73, v8);
      if ( v66 )
        break;
      if ( v65 != v67 )
        _libc_free((unsigned __int64)v65);
      sub_C7D6A0(v63, 16LL * v64, 8);
      if ( v72 != v74 )
        _libc_free((unsigned __int64)v72);
LABEL_30:
      v54 += 84;
      if ( (__int64 *)v52 == v54 )
        return v7;
    }
    v57 = v65;
    v53 = &v65[2 * v66];
    while ( 1 )
    {
      v75 = (unsigned __int64)v77;
      v76 = 0x400000000LL;
      v9 = v54[2];
      v55 = v9 + 160LL * *((unsigned int *)v54 + 6);
      if ( v9 != v55 )
        break;
      if ( !(_DWORD)qword_4FF9E48 )
        goto LABEL_35;
LABEL_24:
      v57 += 2;
      if ( v53 == v57 )
      {
        if ( v65 != v67 )
          _libc_free((unsigned __int64)v65);
        sub_C7D6A0(v63, 16LL * v64, 8);
        if ( v72 != v74 )
          _libc_free((unsigned __int64)v72);
        v7 = 1;
        goto LABEL_30;
      }
    }
    v10 = 0;
    v11 = v54[2];
    do
    {
      v12 = *(__int64 **)v11;
      v59 = *(_QWORD *)v11 + 16LL * *(unsigned int *)(v11 + 8);
      if ( v59 != *(_QWORD *)v11 )
      {
        do
        {
          v13 = v10++;
          v14 = &v72[16 * v13];
          v15 = *(_QWORD *)v14;
          if ( !*(_QWORD *)v14 )
            BUG();
          if ( v66 != 1 )
          {
            if ( !*v57 )
              BUG();
            if ( !(unsigned __int8)sub_B19720(a1[1], *(_QWORD *)(*v57 + 16), *(_QWORD *)(v15 + 16)) )
              goto LABEL_18;
            v15 = *(_QWORD *)v14;
          }
          v16 = (unsigned int)v76;
          v17 = v75;
          v58 = *(_QWORD *)(v11 + 152);
          v18 = v14[8];
          v19 = v76;
          v20 = *(_QWORD *)(v11 + 144);
          v21 = v14[9];
          v22 = *v12;
          v23 = *((unsigned int *)v12 + 2);
          v24 = v75 + 48LL * (unsigned int)v76;
          if ( (unsigned int)v76 >= (unsigned __int64)HIDWORD(v76) )
          {
            v67[2] = v15;
            v27 = (unsigned int)v76 + 1LL;
            v69 = v21;
            v28 = (const __m128i *)v67;
            v67[0] = v20;
            v67[1] = v58;
            v68 = v18;
            v70 = v22;
            v71 = v23;
            if ( HIDWORD(v76) < v27 )
            {
              if ( v75 > (unsigned __int64)v67 || v24 <= (unsigned __int64)v67 )
              {
                sub_C8D5F0((__int64)&v75, v77, v27, 0x30u, HIDWORD(v76), v23);
                v17 = v75;
                v16 = (unsigned int)v76;
              }
              else
              {
                v45 = (char *)v67 - v75;
                sub_C8D5F0((__int64)&v75, v77, v27, 0x30u, HIDWORD(v76), v23);
                v17 = v75;
                v16 = (unsigned int)v76;
                v28 = (const __m128i *)&v45[v75];
              }
            }
            v29 = (__m128i *)(48 * v16 + v17);
            *v29 = _mm_loadu_si128(v28);
            v30 = _mm_loadu_si128(v28 + 1);
            LODWORD(v76) = v76 + 1;
            v29[1] = v30;
            v29[2] = _mm_loadu_si128(v28 + 2);
          }
          else
          {
            if ( v24 )
            {
              *(_QWORD *)(v24 + 16) = v15;
              *(_BYTE *)(v24 + 24) = v18;
              *(_BYTE *)(v24 + 25) = v21;
              *(_QWORD *)(v24 + 32) = v22;
              *(_DWORD *)(v24 + 40) = v23;
              *(_QWORD *)v24 = v20;
              *(_QWORD *)(v24 + 8) = v58;
              v19 = v76;
            }
            LODWORD(v76) = v19 + 1;
          }
LABEL_18:
          v12 += 2;
        }
        while ( (__int64 *)v59 != v12 );
      }
      v11 += 160;
    }
    while ( v55 != v11 );
    if ( (unsigned int)qword_4FF9E48 > (unsigned int)v76 )
      goto LABEL_21;
LABEL_35:
    v31 = v54[1];
    if ( v31 )
    {
      v32 = *(_QWORD *)(v31 + 8);
      v67[0] = (__int64)"const";
      LOWORD(v70) = 259;
      v33 = *((_WORD *)v57 + 4);
      v34 = *v57;
      v35 = sub_BD2C40(72, unk_3F10A14);
      v36 = (__int64)v35;
      if ( v35 )
        sub_B51BF0((__int64)v35, v54[1], v32, (__int64)v67, v34, v33);
    }
    else
    {
      v48 = *(_QWORD *)(*v54 + 8);
      v67[0] = (__int64)"const";
      LOWORD(v70) = 259;
      v49 = *((_WORD *)v57 + 4);
      v50 = *v57;
      v51 = sub_BD2C40(72, unk_3F10A14);
      v36 = (__int64)v51;
      if ( v51 )
        sub_B51BF0((__int64)v51, *v54, v48, (__int64)v67, v50, v49);
    }
    if ( !*v57 )
      BUG();
    v37 = *(_QWORD *)(*v57 + 24);
    v38 = (__int64 *)(v36 + 48);
    v67[0] = v37;
    if ( v37 )
    {
      sub_B96E90((__int64)v67, v37, 1);
      if ( v38 == v67 )
      {
        if ( v67[0] )
          sub_B91220((__int64)v67, v67[0]);
        goto LABEL_43;
      }
      v46 = *(_QWORD *)(v36 + 48);
      if ( !v46 )
      {
LABEL_60:
        v47 = (unsigned __int8 *)v67[0];
        *(_QWORD *)(v36 + 48) = v67[0];
        if ( v47 )
          sub_B976B0((__int64)v67, v47, v36 + 48);
        goto LABEL_43;
      }
    }
    else if ( v38 == v67 || (v46 = *(_QWORD *)(v36 + 48)) == 0 )
    {
LABEL_43:
      v25 = (_BYTE *)v75;
      if ( v75 + 48LL * (unsigned int)v76 == v75 )
      {
LABEL_22:
        if ( v25 != v77 )
          _libc_free((unsigned __int64)v25);
        goto LABEL_24;
      }
      v60 = (_BYTE *)(v75 + 48LL * (unsigned int)v76);
      v39 = v75;
      do
      {
        sub_2736990((__int64)a1, v36, v39);
        v42 = sub_B10CD0(*(_QWORD *)(v39 + 32) + 48LL);
        v43 = sub_B10CD0(v36 + 48);
        v44 = sub_B026B0(v43, v42);
        sub_B10CB0(v67, (__int64)v44);
        if ( v38 == v67 )
        {
          if ( v67[0] )
            sub_B91220(v36 + 48, v67[0]);
        }
        else
        {
          v40 = *(_QWORD *)(v36 + 48);
          if ( v40 )
            sub_B91220(v36 + 48, v40);
          v41 = (unsigned __int8 *)v67[0];
          *(_QWORD *)(v36 + 48) = v67[0];
          if ( v41 )
            sub_B976B0((__int64)v67, v41, v36 + 48);
        }
        v39 += 48;
      }
      while ( v60 != (_BYTE *)v39 );
LABEL_21:
      v25 = (_BYTE *)v75;
      goto LABEL_22;
    }
    sub_B91220(v36 + 48, v46);
    goto LABEL_60;
  }
  return v7;
}
