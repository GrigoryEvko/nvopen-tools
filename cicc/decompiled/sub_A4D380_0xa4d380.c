// Function: sub_A4D380
// Address: 0xa4d380
//
__int64 *__fastcall sub_A4D380(__int64 *a1, __int64 a2)
{
  _QWORD *v4; // rax
  __int64 v5; // rcx
  unsigned __int64 *v6; // rdi
  unsigned __int64 *v7; // r8
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  char v10; // al
  int v11; // r13d
  int v12; // eax
  __int64 v13; // rcx
  char v14; // dl
  char v15; // dl
  volatile signed __int32 *v16; // r13
  signed __int32 v17; // eax
  __int64 v19; // r8
  int v20; // eax
  __int64 v21; // rcx
  char v22; // dl
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // r13
  __m128i *v28; // rax
  __m128i si128; // xmm0
  __int64 v30; // rdx
  unsigned __int64 v31; // rsi
  __int64 v32; // rcx
  unsigned __int64 v33; // r9
  __int64 v34; // r12
  unsigned __int64 v35; // r10
  __int64 v36; // rax
  unsigned __int64 *v37; // rax
  char v38; // al
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 *v41; // rax
  __int64 v42; // rax
  unsigned __int64 *v43; // rax
  signed __int32 v44; // eax
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rax
  __m128i v48; // xmm0
  __int64 *v49; // rsi
  __m128i v50; // xmm0
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r12
  __int64 v54; // rax
  _QWORD *v55; // rax
  unsigned __int64 *v56; // rax
  __int64 v57; // r13
  unsigned __int64 v58; // rdi
  char v59; // [rsp+8h] [rbp-D8h]
  __int64 v60; // [rsp+8h] [rbp-D8h]
  __int64 v61; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v62; // [rsp+10h] [rbp-D0h]
  __int64 *v63; // [rsp+18h] [rbp-C8h]
  int v64; // [rsp+20h] [rbp-C0h]
  __int64 v65; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int64 *v66; // [rsp+40h] [rbp-A0h] BYREF
  volatile signed __int32 *v67; // [rsp+48h] [rbp-98h]
  __int64 v68; // [rsp+50h] [rbp-90h] BYREF
  char v69; // [rsp+58h] [rbp-88h]
  unsigned __int64 *v70; // [rsp+60h] [rbp-80h] BYREF
  char v71; // [rsp+68h] [rbp-78h]
  unsigned __int64 *v72; // [rsp+70h] [rbp-70h] BYREF
  char v73; // [rsp+78h] [rbp-68h]
  unsigned __int64 v74; // [rsp+80h] [rbp-60h] BYREF
  char v75; // [rsp+88h] [rbp-58h]
  __int64 v76; // [rsp+90h] [rbp-50h] BYREF
  unsigned __int64 v77; // [rsp+98h] [rbp-48h]
  _QWORD v78[8]; // [rsp+A0h] [rbp-40h] BYREF

  v66 = 0;
  v4 = (_QWORD *)sub_22077B0(544);
  if ( v4 )
  {
    v4[1] = 0x100000001LL;
    v4[3] = 0x2000000000LL;
    *v4 = &unk_49D9900;
    v4[2] = v4 + 4;
  }
  v67 = (volatile signed __int32 *)v4;
  v6 = (unsigned __int64 *)&v68;
  v66 = v4 + 2;
  sub_9CE2D0((__int64)&v68, a2, 5, v5);
  v8 = v69 & 1;
  v9 = (unsigned int)(2 * v8);
  v10 = (2 * v8) | v69 & 0xFD;
  v69 = v10;
  if ( (_BYTE)v8 )
  {
    v69 = v10 & 0xFD;
    v42 = v68;
    v68 = 0;
    *a1 = v42 | 1;
LABEL_50:
    if ( v68 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v68 + 8LL))(v68);
    goto LABEL_14;
  }
  v64 = v68;
  if ( !(_DWORD)v68 )
  {
LABEL_31:
    v26 = *((unsigned int *)v66 + 2);
    if ( (_DWORD)v26 )
    {
      sub_A4D330(a2 + 40, (__int64 *)&v66);
      *a1 = 1;
    }
    else
    {
      v27 = sub_2241E50(v6, v26, v8, v9, v7);
      v74 = 30;
      v76 = (__int64)v78;
      v28 = (__m128i *)sub_22409D0(&v76, &v74, 0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F23210);
      v76 = (__int64)v28;
      v78[0] = v74;
      qmemcpy(&v28[1], "th no operands", 14);
      *v28 = si128;
      v77 = v74;
      *(_BYTE *)(v76 + v74) = 0;
      sub_C63F00(a1, &v76, 84, v27);
      if ( (_QWORD *)v76 != v78 )
        j_j___libc_free_0(v76, v78[0] + 1LL);
    }
    goto LABEL_12;
  }
  v63 = a1;
  v11 = 0;
  while ( 1 )
  {
    sub_9C66D0((__int64)&v70, a2, 1, v9);
    v12 = v71 & 1;
    v13 = (unsigned int)(2 * v12);
    v14 = (2 * v12) | v71 & 0xFD;
    v71 = v14;
    if ( (_BYTE)v12 )
    {
      a1 = v63;
      v71 = v14 & 0xFD;
      v43 = v70;
      v70 = 0;
      *v63 = (unsigned __int64)v43 | 1;
      goto LABEL_57;
    }
    if ( !v70 )
      break;
    sub_A4B2C0((__int64)&v76, a2, 8, v13);
    v15 = v77 & 1;
    LOBYTE(v77) = (2 * (v77 & 1)) | v77 & 0xFD;
    if ( v15 )
    {
      a1 = v63;
      *v63 = v76 | 1;
      goto LABEL_10;
    }
    v6 = v66;
    v39 = v76;
    v40 = *((unsigned int *)v66 + 2);
    v9 = *((unsigned int *)v66 + 3);
    v8 = v40 + 1;
    if ( v40 + 1 > v9 )
    {
      v60 = v76;
      sub_C8D5F0(v66, v66 + 2, v8, 16);
      v6 = v66;
      v39 = v60;
      v40 = *((unsigned int *)v66 + 2);
    }
    v41 = (__int64 *)(*v6 + 16 * v40);
    *v41 = v39;
    v41[1] = 1;
    ++*((_DWORD *)v6 + 2);
    if ( (v77 & 2) != 0 )
      sub_9CDF70(&v76);
    if ( (v77 & 1) != 0 )
    {
      v6 = (unsigned __int64 *)v76;
      if ( v76 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v76 + 8LL))(v76);
    }
LABEL_27:
    if ( (v71 & 2) != 0 )
      goto LABEL_63;
    if ( (v71 & 1) != 0 )
    {
      v6 = v70;
      if ( v70 )
        (*(void (__fastcall **)(unsigned __int64 *))(*v70 + 8))(v70);
    }
    if ( v64 == ++v11 )
    {
      a1 = v63;
      goto LABEL_31;
    }
  }
  v6 = (unsigned __int64 *)&v72;
  sub_9C66D0((__int64)&v72, a2, 3, v13);
  v20 = v73 & 1;
  v21 = (unsigned int)(2 * v20);
  v22 = (2 * v20) | v73 & 0xFD;
  v73 = v22;
  if ( (_BYTE)v20 )
  {
    a1 = v63;
    v73 = v22 & 0xFD;
    v56 = v72;
    v72 = 0;
    *v63 = (unsigned __int64)v56 | 1;
    goto LABEL_74;
  }
  if ( (unsigned __int64)v72 - 1 > 4 )
  {
    v57 = sub_2241E50(&v72, a2, (char *)v72 - 1, v21, v19);
    a1 = v63;
    v76 = (__int64)v78;
    v74 = 16;
    v76 = sub_22409D0(&v76, &v74, 0);
    v78[0] = v74;
    *(__m128i *)v76 = _mm_load_si128((const __m128i *)&xmmword_3F231D0);
    v77 = v74;
    *(_BYTE *)(v76 + v74) = 0;
    sub_C63F00(v63, &v76, 84, v57);
    if ( (_QWORD *)v76 != v78 )
      j_j___libc_free_0(v76, v78[0] + 1LL);
    goto LABEL_72;
  }
  if ( (unsigned int)((_DWORD)v72 - 3) <= 2 )
  {
    v23 = (__int64)v66;
    v24 = 2LL * ((unsigned __int8)v72 & 7);
    v25 = *((unsigned int *)v66 + 2);
    v9 = *((unsigned int *)v66 + 3);
    v7 = (unsigned __int64 *)(v25 + 1);
    if ( v25 + 1 > v9 )
    {
      v6 = v66;
      v61 = 2LL * ((unsigned __int8)v72 & 7);
      sub_C8D5F0(v66, v66 + 2, v25 + 1, 16);
      v25 = *((unsigned int *)v66 + 2);
      v24 = v61;
    }
    v8 = *v66 + 16 * v25;
    *(_QWORD *)v8 = 0;
    *(_QWORD *)(v8 + 8) = v24;
    ++*(_DWORD *)(v23 + 8);
LABEL_25:
    if ( (v73 & 2) != 0 )
      goto LABEL_66;
    if ( (v73 & 1) != 0 )
    {
      v6 = v72;
      if ( v72 )
        (*(void (__fastcall **)(unsigned __int64 *))(*v72 + 8))(v72);
    }
    goto LABEL_27;
  }
  v59 = (char)v72;
  v6 = &v74;
  sub_A4B2C0((__int64)&v74, a2, 5, v21);
  v7 = &v74;
  v30 = v75 & 1;
  v31 = (unsigned int)(2 * v30);
  v32 = (unsigned int)v31 | v75 & 0xFD;
  v75 = (2 * v30) | v75 & 0xFD;
  if ( (_BYTE)v30 )
  {
    v58 = v74;
    a1 = v63;
    v74 = 0;
    v75 = v32 & 0xFD;
    *v63 = v58 | 1;
    goto LABEL_72;
  }
  v33 = v74;
  if ( !v74 )
  {
    v53 = (__int64)v66;
    v54 = *((unsigned int *)v66 + 2);
    v9 = *((unsigned int *)v66 + 3);
    v8 = v54 + 1;
    if ( v54 + 1 > v9 )
    {
      v31 = (unsigned __int64)(v66 + 2);
      v6 = v66;
      sub_C8D5F0(v66, v66 + 2, v8, 16);
      v54 = *((unsigned int *)v66 + 2);
      v7 = &v74;
    }
    v55 = (_QWORD *)(*v66 + 16 * v54);
    *v55 = 0;
    v55[1] = 1;
    ++*(_DWORD *)(v53 + 8);
    v38 = v75;
    if ( (v75 & 2) != 0 )
      goto LABEL_80;
    goto LABEL_40;
  }
  if ( v74 <= 0x20 )
  {
    v34 = (__int64)v66;
    v35 = (unsigned __int8)(2 * (v59 & 7));
    v36 = *((unsigned int *)v66 + 2);
    v9 = *((unsigned int *)v66 + 3);
    v8 = v36 + 1;
    if ( v36 + 1 > v9 )
    {
      v31 = (unsigned __int64)(v66 + 2);
      v6 = v66;
      v62 = v74;
      sub_C8D5F0(v66, v66 + 2, v8, 16);
      v36 = *((unsigned int *)v66 + 2);
      v7 = &v74;
      v35 = (unsigned __int8)(2 * (v59 & 7));
      v33 = v62;
    }
    v37 = (unsigned __int64 *)(*v66 + 16 * v36);
    *v37 = v33;
    v37[1] = v35;
    ++*(_DWORD *)(v34 + 8);
    v38 = v75;
    if ( (v75 & 2) != 0 )
      goto LABEL_80;
LABEL_40:
    if ( (v38 & 1) != 0 )
    {
      v6 = (unsigned __int64 *)v74;
      if ( v74 )
        (*(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64, unsigned __int64, unsigned __int64 *))(*(_QWORD *)v74 + 8LL))(
          v74,
          v31,
          v8,
          v9,
          &v74);
    }
    goto LABEL_25;
  }
  a1 = v63;
  v45 = sub_2241E50(&v74, v31, v30, v32, &v74);
  v76 = (__int64)v78;
  v46 = v45;
  v65 = 51;
  v47 = sub_22409D0(&v76, &v65, 0);
  v76 = v47;
  v78[0] = v65;
  *(__m128i *)v47 = _mm_load_si128((const __m128i *)&xmmword_3F231E0);
  v48 = _mm_load_si128((const __m128i *)&xmmword_3F231F0);
  v49 = &v76;
  *(_WORD *)(v47 + 48) = 29793;
  *(__m128i *)(v47 + 16) = v48;
  v50 = _mm_load_si128((const __m128i *)&xmmword_3F23200);
  *(_BYTE *)(v47 + 50) = 97;
  *(__m128i *)(v47 + 32) = v50;
  v77 = v65;
  *(_BYTE *)(v76 + v65) = 0;
  sub_C63F00(v63, &v76, 84, v46);
  if ( (_QWORD *)v76 != v78 )
  {
    v49 = (__int64 *)(v78[0] + 1LL);
    j_j___libc_free_0(v76, v78[0] + 1LL);
  }
  if ( (v75 & 2) != 0 )
LABEL_80:
    sub_9CDF70(&v74);
  if ( (v75 & 1) != 0 && v74 )
    (*(void (__fastcall **)(unsigned __int64, __int64 *, __int64, __int64, unsigned __int64 *))(*(_QWORD *)v74 + 8LL))(
      v74,
      v49,
      v51,
      v52,
      &v74);
LABEL_72:
  if ( (v73 & 2) != 0 )
LABEL_66:
    sub_9CDF70(&v72);
  if ( (v73 & 1) != 0 )
  {
LABEL_74:
    if ( v72 )
      (*(void (__fastcall **)(unsigned __int64 *))(*v72 + 8))(v72);
  }
LABEL_10:
  if ( (v71 & 2) != 0 )
LABEL_63:
    sub_9CDF70(&v70);
  if ( (v71 & 1) != 0 )
  {
LABEL_57:
    if ( v70 )
      (*(void (__fastcall **)(unsigned __int64 *))(*v70 + 8))(v70);
  }
LABEL_12:
  if ( (v69 & 2) != 0 )
    sub_9CE230(&v68);
  if ( (v69 & 1) != 0 )
    goto LABEL_50;
LABEL_14:
  v16 = v67;
  if ( v67 )
  {
    if ( &_pthread_key_create )
    {
      v17 = _InterlockedExchangeAdd(v67 + 2, 0xFFFFFFFF);
    }
    else
    {
      v17 = *((_DWORD *)v67 + 2);
      *((_DWORD *)v67 + 2) = v17 - 1;
    }
    if ( v17 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 16LL))(v16);
      if ( &_pthread_key_create )
      {
        v44 = _InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
      }
      else
      {
        v44 = *((_DWORD *)v16 + 3);
        *((_DWORD *)v16 + 3) = v44 - 1;
      }
      if ( v44 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 24LL))(v16);
    }
  }
  return a1;
}
