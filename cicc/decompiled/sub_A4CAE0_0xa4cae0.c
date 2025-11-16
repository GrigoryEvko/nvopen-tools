// Function: sub_A4CAE0
// Address: 0xa4cae0
//
__int64 __fastcall sub_A4CAE0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  _DWORD **v8; // r13
  unsigned __int8 v9; // dl
  __int64 v10; // r15
  __int64 v11; // rdx
  unsigned __int8 v12; // cl
  __int64 v13; // rcx
  char v14; // cl
  char v15; // dl
  unsigned int v16; // ecx
  int v17; // edx
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  char v22; // al
  __int64 v23; // r13
  __int64 v24; // rax
  __m128i si128; // xmm0
  __int64 v26; // rax
  int v28; // edx
  __int64 v29; // rcx
  unsigned __int8 v30; // dl
  int v31; // r15d
  __int64 v32; // rax
  char v33; // al
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rax
  __m128i v37; // xmm0
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r8
  int v42; // ecx
  __int64 v43; // rdi
  int v44; // r9d
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // r14
  __int64 v48; // rax
  __m128i v49; // xmm0
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdi
  __int64 v53; // rdi
  char v54; // al
  char v55; // cl
  unsigned __int64 v56; // rdx
  unsigned __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rax
  int v61; // [rsp+Ch] [rbp-94h]
  __int64 v62; // [rsp+10h] [rbp-90h]
  int v63; // [rsp+18h] [rbp-88h]
  int v64; // [rsp+20h] [rbp-80h]
  unsigned int v65; // [rsp+28h] [rbp-78h]
  int v66; // [rsp+28h] [rbp-78h]
  __int64 v67; // [rsp+30h] [rbp-70h] BYREF
  char v68; // [rsp+38h] [rbp-68h]
  __int64 v69; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int8 v70; // [rsp+48h] [rbp-58h]
  __int64 v71; // [rsp+50h] [rbp-50h] BYREF
  __int64 v72; // [rsp+58h] [rbp-48h]
  _QWORD v73[8]; // [rsp+60h] [rbp-40h] BYREF

  if ( a3 == 3 )
  {
    sub_9CE2D0((__int64)&v67, a2, 6, a4);
    v28 = v68 & 1;
    v68 = (2 * v28) | v68 & 0xFD;
    if ( (_BYTE)v28 )
    {
      v39 = v67;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v39 & 0xFFFFFFFFFFFFFFFELL;
      return a1;
    }
    v64 = v67;
    sub_9CE2D0((__int64)&v69, a2, 6, (unsigned int)(2 * v28));
    v29 = v70 & 1;
    v30 = v70 & 0xFD | (2 * v29);
    v70 = v30;
    if ( (_BYTE)v29 )
    {
      v52 = v69;
      *(_BYTE *)(a1 + 8) |= 3u;
      v70 = v30 & 0xFD;
      v69 = 0;
      *(_QWORD *)a1 = v52 & 0xFFFFFFFFFFFFFFFELL;
    }
    else
    {
      v31 = 0;
      v66 = v69;
      if ( (_DWORD)v69 )
      {
        while ( 1 )
        {
          sub_A4B2C0((__int64)&v71, a2, 6, v29);
          if ( (v72 & 1) != 0 )
            break;
          if ( v66 == ++v31 )
          {
            v30 = v70;
            v54 = v70 >> 1;
            v55 = *(_BYTE *)(a1 + 8) & 0xFC | 2;
            *(_DWORD *)a1 = v64;
            *(_BYTE *)(a1 + 8) = v55;
            v33 = v54 & 1;
            goto LABEL_33;
          }
        }
        v32 = v71;
        v30 = v70;
        *(_BYTE *)(a1 + 8) |= 3u;
        *(_QWORD *)a1 = v32 & 0xFFFFFFFFFFFFFFFELL;
        v33 = (v30 & 2) != 0;
LABEL_33:
        if ( v33 )
          sub_9CE230(&v69);
      }
      else
      {
        *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
        *(_DWORD *)a1 = v64;
      }
      if ( (v30 & 1) != 0 && v69 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v69 + 8LL))(v69);
    }
    if ( (v68 & 2) != 0 )
      sub_9CE230(&v67);
    if ( (v68 & 1) != 0 )
    {
      v51 = v67;
      if ( v67 )
        goto LABEL_56;
    }
    return a1;
  }
  v6 = *(_QWORD *)(a2 + 40);
  v7 = (unsigned int)(a3 - 4);
  if ( v7 >= (*(_QWORD *)(a2 + 48) - v6) >> 4 )
  {
    v23 = sub_2241E50(a1, a2, v7, v6, a5);
    v71 = (__int64)v73;
    v69 = 21;
    v24 = sub_22409D0(&v71, &v69, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F23140);
    v71 = v24;
    v73[0] = v69;
    *(_DWORD *)(v24 + 16) = 1700949365;
    *(_BYTE *)(v24 + 20) = 114;
    *(__m128i *)v24 = si128;
    v72 = v69;
    *(_BYTE *)(v71 + v69) = 0;
    sub_C63F00(&v69, &v71, 84, v23);
    if ( (_QWORD *)v71 != v73 )
      j_j___libc_free_0(v71, v73[0] + 1LL);
    v26 = v69;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v26 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  v8 = *(_DWORD ***)(v6 + 16 * v7);
  v9 = *((_BYTE *)*v8 + 8);
  if ( (v9 & 1) != 0 )
  {
    v63 = **v8;
  }
  else
  {
    v34 = (unsigned __int8)(((v9 >> 1) & 7) - 3) & 0xFD;
    if ( !(_DWORD)v34 )
    {
      v35 = sub_2241E50(a1, a2, v34, v6, a5);
      v71 = (__int64)v73;
      v69 = 43;
      v36 = sub_22409D0(&v71, &v69, 0);
      v71 = v36;
      v73[0] = v69;
      *(__m128i *)v36 = _mm_load_si128((const __m128i *)&xmmword_3F23150);
      v37 = _mm_load_si128((const __m128i *)&xmmword_3F23160);
      qmemcpy((void *)(v36 + 32), "y or a Blob", 11);
      *(__m128i *)(v36 + 16) = v37;
      v72 = v69;
      *(_BYTE *)(v71 + v69) = 0;
      sub_C63F00(&v69, &v71, 84, v35);
      if ( (_QWORD *)v71 != v73 )
        j_j___libc_free_0(v71, v73[0] + 1LL);
      v38 = v69;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v38 & 0xFFFFFFFFFFFFFFFELL;
      return a1;
    }
    sub_A4B540((__int64)&v71, a2, (__int64)*v8, v6);
    if ( (v72 & 1) != 0 )
    {
LABEL_43:
      v40 = v71;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v40 & 0xFFFFFFFFFFFFFFFELL;
      return a1;
    }
    v63 = v71;
  }
  v65 = *((_DWORD *)v8 + 2);
  if ( v65 <= 1 )
  {
LABEL_25:
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_DWORD *)a1 = v63;
    return a1;
  }
  LODWORD(v10) = 1;
  while ( 1 )
  {
    v11 = (__int64)&(*v8)[4 * (unsigned int)v10];
    v12 = *(_BYTE *)(v11 + 8);
    if ( (v12 & 1) != 0 )
      goto LABEL_24;
    v13 = (v12 >> 1) & 7;
    if ( (_DWORD)v13 != 3 )
    {
      if ( (_DWORD)v13 == 5 )
      {
        sub_9CE2D0((__int64)&v71, a2, 6, v13);
        v14 = v72 & 1;
        v15 = (2 * (v72 & 1)) | v72 & 0xFD;
        LOBYTE(v72) = v15;
        if ( v14 )
        {
          v58 = v71;
          *(_BYTE *)(a1 + 8) |= 3u;
          LOBYTE(v72) = v15 & 0xFD;
          v71 = 0;
          *(_QWORD *)a1 = v58 & 0xFFFFFFFFFFFFFFFELL;
LABEL_83:
          v51 = v71;
          if ( v71 )
            goto LABEL_56;
          return a1;
        }
        v16 = *(_DWORD *)(a2 + 32);
        v17 = v71;
        if ( v16 > 0x1F )
        {
          *(_DWORD *)(a2 + 32) = 32;
          *(_QWORD *)(a2 + 24) >>= (unsigned __int8)v16 - 32;
          v18 = 32;
        }
        else
        {
          *(_DWORD *)(a2 + 32) = 0;
          v18 = 0;
        }
        v19 = 32 * ((v17 != 0) + ((v17 - (unsigned int)(v17 != 0)) >> 2)) + 8LL * *(_QWORD *)(a2 + 16) - v18;
        v20 = *(_QWORD *)(a2 + 8);
        if ( v19 >> 3 > v20 )
        {
          *(_QWORD *)(a2 + 16) = v20;
          goto LABEL_25;
        }
        sub_9CDFE0(&v69, a2, v19, v20);
        v21 = v69 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v69 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v22 = v72;
          *(_BYTE *)(a1 + 8) |= 3u;
          *(_QWORD *)a1 = v21;
          if ( (v22 & 2) != 0 )
LABEL_79:
            sub_9CE230(&v71);
          if ( (v22 & 1) == 0 )
            return a1;
          goto LABEL_83;
        }
        if ( (v72 & 2) != 0 )
          goto LABEL_79;
        if ( (v72 & 1) != 0 )
        {
          v53 = v71;
          if ( v71 )
            goto LABEL_65;
        }
      }
      else
      {
        sub_A4B540((__int64)&v71, a2, v11, v13);
        if ( (v72 & 1) != 0 )
          goto LABEL_43;
      }
      goto LABEL_24;
    }
    sub_9CE2D0((__int64)&v69, a2, 6, v13);
    v42 = v70 & 1;
    v43 = (unsigned int)(2 * v42);
    v70 = (2 * v42) | v70 & 0xFD;
    if ( (_BYTE)v42 )
    {
      v59 = v69;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v59 & 0xFFFFFFFFFFFFFFFELL;
      return a1;
    }
    v44 = v69;
    v10 = (unsigned int)(v10 + 1);
    v45 = (__int64)&(*v8)[4 * v10];
    v46 = (*(_BYTE *)(v45 + 8) >> 1) & 7;
    if ( (_BYTE)v46 == 2 )
      break;
    if ( (_BYTE)v46 == 4 )
    {
      v45 = *(unsigned int *)(a2 + 32);
      v56 = (unsigned int)(6 * v69) + 8LL * *(_QWORD *)(a2 + 16) - v45;
    }
    else
    {
      if ( (_BYTE)v46 != 1 )
      {
        v47 = sub_2241E50(v43, a2, v46, v45, v41);
        v71 = (__int64)v73;
        v67 = 46;
        v48 = sub_22409D0(&v71, &v67, 0);
        v71 = v48;
        v73[0] = v67;
        *(__m128i *)v48 = _mm_load_si128((const __m128i *)&xmmword_3F23180);
        v49 = _mm_load_si128((const __m128i *)&xmmword_3F231B0);
        qmemcpy((void *)(v48 + 32), "rray or a Blob", 14);
        *(__m128i *)(v48 + 16) = v49;
        v72 = v67;
        *(_BYTE *)(v71 + v67) = 0;
        sub_C63F00(&v67, &v71, 84, v47);
        if ( (_QWORD *)v71 != v73 )
          j_j___libc_free_0(v71, v73[0] + 1LL);
        v50 = v67;
        *(_BYTE *)(a1 + 8) |= 3u;
        *(_QWORD *)a1 = v50 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_53;
      }
      v56 = 8LL * *(_QWORD *)(a2 + 16) - *(unsigned int *)(a2 + 32) + *(_QWORD *)v45 * (unsigned int)v69;
    }
    sub_9CDFE0(&v71, a2, v56, v45);
    v57 = v71 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v71 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v57;
      goto LABEL_53;
    }
LABEL_69:
    if ( (v70 & 2) != 0 )
      goto LABEL_81;
    if ( (v70 & 1) != 0 )
    {
      v53 = v69;
      if ( v69 )
LABEL_65:
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v53 + 8LL))(v53);
    }
LABEL_24:
    LODWORD(v10) = v10 + 1;
    if ( (unsigned int)v10 >= v65 )
      goto LABEL_25;
  }
  if ( !(_DWORD)v69 )
    goto LABEL_24;
  while ( 1 )
  {
    v61 = v44;
    v62 = v45;
    sub_A4B2C0((__int64)&v71, a2, *(unsigned int *)v45, v45);
    v45 = v62;
    if ( (v72 & 1) != 0 )
      break;
    v44 = v61 - 1;
    if ( v61 == 1 )
      goto LABEL_69;
  }
  v60 = v71;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v60 & 0xFFFFFFFFFFFFFFFELL;
LABEL_53:
  if ( (v70 & 2) != 0 )
LABEL_81:
    sub_9CE230(&v69);
  if ( (v70 & 1) == 0 )
    return a1;
  v51 = v69;
  if ( !v69 )
    return a1;
LABEL_56:
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v51 + 8LL))(v51);
  return a1;
}
