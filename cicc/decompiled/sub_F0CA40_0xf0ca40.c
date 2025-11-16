// Function: sub_F0CA40
// Address: 0xf0ca40
//
unsigned __int8 *__fastcall sub_F0CA40(
        __int64 a1,
        unsigned __int8 *a2,
        const __m128i *a3,
        __int64 *a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  int v12; // r14d
  unsigned __int8 v13; // r15
  __m128i v14; // xmm5
  unsigned __int64 v15; // xmm6_8
  __m128i v16; // xmm7
  __int64 v17; // rax
  __int64 v18; // r14
  unsigned __int8 *v19; // r15
  __int64 v21; // rcx
  __m128i v22; // xmm1
  unsigned __int64 v23; // xmm2_8
  __m128i v24; // xmm3
  __int64 v25; // rax
  __int64 v26; // rdi
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  char v29; // bl
  char v30; // r13
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int8 v33; // dl
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // rsi
  unsigned int v38; // r12d
  __int64 v39; // rax
  __int64 v40; // rax
  __m128i v41; // rax
  __int64 v42; // rdx
  unsigned __int8 v43; // dl
  __int64 v44; // rdx
  _BYTE *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __m128i v48; // rax
  __int64 v49; // rdx
  int v50; // r8d
  __int64 v51; // r14
  __int64 v52; // rbx
  __int64 v53; // rdx
  unsigned int v54; // esi
  const __m128i *v55; // [rsp+0h] [rbp-F0h]
  int v56; // [rsp+8h] [rbp-E8h]
  __int64 v57; // [rsp+8h] [rbp-E8h]
  bool v58; // [rsp+10h] [rbp-E0h]
  unsigned __int8 *v59; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v60; // [rsp+20h] [rbp-D0h]
  unsigned int v62; // [rsp+34h] [rbp-BCh]
  int v64[8]; // [rsp+40h] [rbp-B0h] BYREF
  __int16 v65; // [rsp+60h] [rbp-90h]
  __m128i v66; // [rsp+70h] [rbp-80h] BYREF
  __m128i v67; // [rsp+80h] [rbp-70h]
  unsigned __int64 v68; // [rsp+90h] [rbp-60h]
  unsigned __int8 *v69; // [rsp+98h] [rbp-58h]
  __m128i v70; // [rsp+A0h] [rbp-50h]
  __int64 v71; // [rsp+B0h] [rbp-40h]

  v12 = *a2;
  v59 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
  v13 = *a2;
  v60 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v62 = v12 - 29;
  if ( (unsigned __int8)sub_DF9AB0(*(_QWORD *)(a1 + 8)) && (_BYTE)qword_4F8B2E8 && *a2 == 42 && sub_F06F10((__int64)a2) )
    return 0;
  v58 = a5 <= 0x1E && ((1LL << a5) & 0x70066000) != 0;
  if ( a5 == 28 )
  {
    if ( (unsigned int)(v12 - 58) > 1 )
      goto LABEL_8;
  }
  else if ( a5 == 29 )
  {
    if ( v12 != 57 )
      goto LABEL_8;
  }
  else if ( a5 != 17 || ((v13 - 42) & 0xFD) != 0 )
  {
    goto LABEL_8;
  }
  if ( a6 == a8 )
    goto LABEL_25;
  if ( a6 == a9 && v58 )
  {
    v21 = a9;
    a9 = a8;
    a8 = v21;
LABEL_25:
    v22 = _mm_loadu_si128(a3 + 1);
    v23 = _mm_loadu_si128(a3 + 2).m128i_u64[0];
    v24 = _mm_loadu_si128(a3 + 3);
    v25 = a3[4].m128i_i64[0];
    v66 = _mm_loadu_si128(a3);
    v68 = v23;
    v71 = v25;
    v69 = a2;
    v67 = v22;
    v70 = v24;
    v18 = sub_101E7C0(v62, a7, a9, &v66);
    if ( v18
      || ((v39 = *((_QWORD *)v59 + 2)) != 0 && !*(_QWORD *)(v39 + 8)
       || (v40 = *((_QWORD *)v60 + 2)) != 0 && !*(_QWORD *)(v40 + 8))
      && (v41.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v60),
          v66 = v41,
          LOWORD(v68) = 261,
          (v18 = sub_F0A990(a4, v62, a7, a9, v64[0], 0, (__int64)&v66, 0)) != 0) )
    {
      v26 = a4[10];
      v65 = 257;
      v19 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v26 + 16LL))(
                                 v26,
                                 a5,
                                 a6,
                                 v18);
      if ( v19 )
        goto LABEL_27;
      LOWORD(v68) = 257;
      v19 = (unsigned __int8 *)sub_B504D0(a5, a6, v18, (__int64)&v66, 0, 0);
      if ( (unsigned __int8)sub_920620((__int64)v19) )
      {
        v49 = a4[12];
        v50 = *((_DWORD *)a4 + 26);
        if ( v49 )
        {
          v56 = *((_DWORD *)a4 + 26);
          sub_B99FD0((__int64)v19, 3u, v49);
          v50 = v56;
        }
        sub_B45150((__int64)v19, v50);
      }
      (*(void (__fastcall **)(__int64, unsigned __int8 *, int *, __int64, __int64))(*(_QWORD *)a4[11] + 16LL))(
        a4[11],
        v19,
        v64,
        a4[7],
        a4[8]);
      if ( *a4 != *a4 + 16LL * *((unsigned int *)a4 + 2) )
      {
        v57 = v18;
        v51 = *a4 + 16LL * *((unsigned int *)a4 + 2);
        v55 = a3;
        v52 = *a4;
        do
        {
          v53 = *(_QWORD *)(v52 + 8);
          v54 = *(_DWORD *)v52;
          v52 += 16;
          sub_B99FD0((__int64)v19, v54, v53);
        }
        while ( v51 != v52 );
        v18 = v57;
        a3 = v55;
      }
      if ( v19 )
        goto LABEL_27;
    }
  }
LABEL_8:
  if ( !sub_F075A0(v62, a5) )
    return 0;
  if ( a7 == a9 )
    goto LABEL_13;
  if ( a7 != a8 || !v58 )
    return 0;
  a8 = a9;
LABEL_13:
  v14 = _mm_loadu_si128(a3 + 1);
  v15 = _mm_loadu_si128(a3 + 2).m128i_u64[0];
  v16 = _mm_loadu_si128(a3 + 3);
  v17 = a3[4].m128i_i64[0];
  v66 = _mm_loadu_si128(a3);
  v68 = v15;
  v71 = v17;
  v69 = a2;
  v67 = v14;
  v70 = v16;
  v18 = sub_101E7C0(v62, a6, a8, &v66);
  if ( !v18 )
  {
    v46 = *((_QWORD *)v59 + 2);
    if ( !v46 || *(_QWORD *)(v46 + 8) )
    {
      v47 = *((_QWORD *)v60 + 2);
      if ( !v47 || *(_QWORD *)(v47 + 8) )
        return 0;
    }
    v48.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v59);
    LOWORD(v68) = 261;
    v66 = v48;
    v18 = sub_F0A990(a4, v62, a6, a8, v64[0], 0, (__int64)&v66, 0);
    if ( !v18 )
      return 0;
  }
  LOWORD(v68) = 257;
  v19 = (unsigned __int8 *)sub_F0A990(a4, a5, v18, a7, v64[0], 0, (__int64)&v66, 0);
  if ( !v19 )
    return 0;
LABEL_27:
  sub_BD6B90(v19, a2);
  if ( (unsigned __int8)(*v19 - 42) > 0x11u )
    return v19;
  v27 = *a2;
  if ( (unsigned __int8)v27 <= 0x36u && (v28 = 0x40540000000000LL, _bittest64(&v28, v27)) )
  {
    v29 = sub_B44900((__int64)a2);
    v30 = sub_B448F0((__int64)a2);
  }
  else
  {
    v30 = 0;
    v29 = 0;
  }
  v31 = *v59;
  if ( (unsigned __int8)v31 <= 0x1Cu )
  {
    if ( (_BYTE)v31 == 5 && ((*((_WORD *)v59 + 1) & 0xFFF7) == 0x11 || (*((_WORD *)v59 + 1) & 0xFFFD) == 0xD) )
      goto LABEL_34;
  }
  else if ( (unsigned __int8)v31 <= 0x36u )
  {
    v32 = 0x40540000000000LL;
    if ( _bittest64(&v32, v31) )
    {
LABEL_34:
      v33 = v59[1];
      v29 &= (v33 & 4) != 0;
      v30 &= (v33 & 2) != 0;
    }
  }
  v34 = *v60;
  if ( (unsigned __int8)v34 <= 0x1Cu )
  {
    if ( (_BYTE)v34 == 5 && ((*((_WORD *)v60 + 1) & 0xFFF7) == 0x11 || (*((_WORD *)v60 + 1) & 0xFFFD) == 0xD) )
      goto LABEL_65;
  }
  else if ( (unsigned __int8)v34 <= 0x36u )
  {
    v42 = 0x40540000000000LL;
    if ( _bittest64(&v42, v34) )
    {
LABEL_65:
      v43 = v60[1];
      v29 &= (v43 & 4) != 0;
      v30 &= (v43 & 2) != 0;
    }
  }
  if ( v62 == 13 && a5 == 17 )
  {
    v35 = v18 + 24;
    if ( *(_BYTE *)v18 != 17 )
    {
      v44 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v18 + 8) + 8LL) - 17;
      if ( (unsigned int)v44 > 1 )
        goto LABEL_44;
      if ( *(_BYTE *)v18 > 0x15u )
        goto LABEL_44;
      v45 = sub_AD7630(v18, 0, v44);
      if ( !v45 || *v45 != 17 )
        goto LABEL_44;
      v35 = (__int64)(v45 + 24);
    }
    v36 = *(_DWORD *)(v35 + 8);
    v37 = *(_QWORD *)v35;
    v38 = v36 - 1;
    if ( v36 <= 0x40 )
    {
      if ( v37 == 1LL << v38 )
        goto LABEL_44;
    }
    else if ( (*(_QWORD *)(v37 + 8LL * (v38 >> 6)) & (1LL << v38)) != 0 && v38 == (unsigned int)sub_C44590(v35) )
    {
      goto LABEL_44;
    }
    sub_B44850(v19, v29);
LABEL_44:
    sub_B447F0(v19, v30);
  }
  return v19;
}
