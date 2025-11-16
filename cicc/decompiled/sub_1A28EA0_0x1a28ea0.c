// Function: sub_1A28EA0
// Address: 0x1a28ea0
//
void __fastcall sub_1A28EA0(unsigned __int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  __int64 *v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  _QWORD *v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // ebx
  __int64 v24; // rax
  __int64 v25; // r13
  _BYTE *v26; // rdx
  __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rdi
  _BYTE *v32; // rbx
  unsigned __int64 v33; // r12
  __int64 v34; // rdi
  __int64 *v35; // r13
  __int64 v36; // rax
  const __m128i *v37; // rsi
  signed __int64 v38; // rax
  __int64 *v39; // rdi
  const __m128i *v40; // rax
  size_t v41; // rdx
  __int64 *v42; // rbx
  unsigned int v43; // eax
  __int64 v44; // rbx
  __int64 *v45; // r12
  unsigned __int64 v46; // rax
  unsigned __int64 *v47; // rbx
  unsigned __int64 *v48; // rdi
  _BYTE *v49; // rbx
  __int64 v50; // rdi
  __int64 v51; // [rsp+8h] [rbp-2B8h]
  __int64 v52; // [rsp+10h] [rbp-2B0h] BYREF
  unsigned int v53; // [rsp+18h] [rbp-2A8h]
  __int64 v54; // [rsp+20h] [rbp-2A0h] BYREF
  __int64 v55; // [rsp+28h] [rbp-298h]
  __int64 v56; // [rsp+30h] [rbp-290h]
  _BYTE *v57; // [rsp+38h] [rbp-288h]
  __int64 v58; // [rsp+40h] [rbp-280h]
  _BYTE v59[192]; // [rsp+48h] [rbp-278h] BYREF
  __int64 v60; // [rsp+108h] [rbp-1B8h]
  _BYTE *v61; // [rsp+110h] [rbp-1B0h]
  _BYTE *v62; // [rsp+118h] [rbp-1A8h]
  __int64 v63; // [rsp+120h] [rbp-1A0h]
  int v64; // [rsp+128h] [rbp-198h]
  _BYTE v65[64]; // [rsp+130h] [rbp-190h] BYREF
  unsigned __int64 v66; // [rsp+170h] [rbp-150h]
  char v67; // [rsp+178h] [rbp-148h]
  __int64 v68; // [rsp+180h] [rbp-140h]
  unsigned int v69; // [rsp+188h] [rbp-138h]
  unsigned __int64 v70; // [rsp+190h] [rbp-130h]
  unsigned __int64 *v71; // [rsp+198h] [rbp-128h]
  __int64 v72; // [rsp+1A0h] [rbp-120h]
  __int64 v73; // [rsp+1A8h] [rbp-118h]
  __int64 v74; // [rsp+1B0h] [rbp-110h] BYREF
  __int64 v75; // [rsp+1F0h] [rbp-D0h] BYREF
  __int64 v76; // [rsp+1F8h] [rbp-C8h]
  __int64 v77; // [rsp+200h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+240h] [rbp-80h] BYREF
  _BYTE *v79; // [rsp+248h] [rbp-78h]
  _BYTE *v80; // [rsp+250h] [rbp-70h]
  __int64 v81; // [rsp+258h] [rbp-68h]
  int v82; // [rsp+260h] [rbp-60h]
  _BYTE v83[88]; // [rsp+268h] [rbp-58h] BYREF

  a1[27] = (unsigned __int64)(a1 + 29);
  a1[1] = (unsigned __int64)(a1 + 3);
  a1[37] = (unsigned __int64)(a1 + 39);
  a1[2] = 0x800000000LL;
  a1[28] = 0x800000000LL;
  a1[38] = 0x800000000LL;
  a1[47] = (unsigned __int64)(a1 + 49);
  a1[48] = 0x800000000LL;
  *a1 = 0;
  v54 = a2;
  v6 = a3[7];
  v58 = 0x800000000LL;
  v55 = 0;
  v56 = 0;
  v57 = v59;
  v60 = 0;
  v61 = v65;
  v62 = v65;
  v63 = 8;
  v64 = 0;
  v69 = 1;
  v68 = 0;
  v7 = sub_12BE0A0(a2, v6);
  v71 = a1;
  v72 = 0;
  v73 = 1;
  v70 = v7;
  v8 = &v74;
  do
  {
    *v8 = -8;
    v8 += 2;
  }
  while ( v8 != &v75 );
  v75 = 0;
  v9 = &v77;
  v76 = 1;
  do
  {
    *v9 = -8;
    v9 += 2;
  }
  while ( v9 != &v78 );
  v10 = *a3;
  v78 = 0;
  v79 = v83;
  v80 = v83;
  v81 = 4;
  v82 = 0;
  v11 = sub_15A9650(v54, v10);
  v67 = 1;
  v53 = *(_DWORD *)(v11 + 8) >> 8;
  if ( v53 > 0x40 )
    sub_16A4EF0((__int64)&v52, 0, 0);
  else
    v52 = 0;
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0(v68);
  v55 &= 3u;
  v56 &= 3u;
  v68 = v52;
  v69 = v53;
  sub_386EA80(&v54, a3);
  v15 = (unsigned int)v58;
  if ( !(_DWORD)v58 )
    goto LABEL_28;
  while ( 1 )
  {
    v22 = (__int64)&v57[24 * v15 - 24];
    v23 = *(_DWORD *)(v22 + 16);
    v24 = *(_QWORD *)v22;
    *(_DWORD *)(v22 + 16) = 0;
    v25 = *(_QWORD *)(v22 + 8);
    LODWORD(v58) = v58 - 1;
    v26 = &v57[24 * (unsigned int)v58];
    if ( *((_DWORD *)v26 + 4) > 0x40u )
    {
      v27 = *((_QWORD *)v26 + 1);
      if ( v27 )
      {
        v51 = v24;
        j_j___libc_free_0_0(v27);
        v24 = v51;
      }
    }
    v28 = v24 & 0xFFFFFFFFFFFFFFF8LL;
    v66 = v24 & 0xFFFFFFFFFFFFFFF8LL;
    v67 = (v24 >> 2) & 1;
    if ( v67 )
    {
      if ( v69 > 0x40 && v68 )
      {
        j_j___libc_free_0_0(v68);
        v28 = v66;
      }
      v69 = v23;
      v23 = 0;
      v68 = v25;
    }
    v16 = sub_1648700(v28);
    sub_1A28BE0((__int64)&v54, (__int64)v16, v17, v18, v19, v20);
    v21 = v55;
    if ( (v55 & 4) != 0 )
      break;
    if ( v23 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    v15 = (unsigned int)v58;
    if ( !(_DWORD)v58 )
      goto LABEL_28;
  }
  if ( v23 > 0x40 )
  {
    if ( v25 )
      j_j___libc_free_0_0(v25);
LABEL_28:
    v21 = v55;
  }
  if ( (v56 & 4) != 0 || (v21 & 4) != 0 )
  {
    v29 = v21 & 0xFFFFFFFFFFFFFFF8LL;
    v30 = v56 & 0xFFFFFFFFFFFFFFF8LL;
    v31 = (unsigned __int64)v80;
    if ( (v56 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      v30 = v29;
    *a1 = v30;
    if ( (_BYTE *)v31 != v79 )
      _libc_free(v31);
    if ( (v76 & 1) != 0 )
    {
      if ( (v73 & 1) != 0 )
        goto LABEL_37;
    }
    else
    {
      j___libc_free_0(v77);
      if ( (v73 & 1) != 0 )
      {
LABEL_37:
        if ( v69 <= 0x40 )
          goto LABEL_40;
LABEL_38:
        if ( v68 )
          j_j___libc_free_0_0(v68);
LABEL_40:
        if ( v62 != v61 )
          _libc_free((unsigned __int64)v62);
        v32 = v57;
        v33 = (unsigned __int64)&v57[24 * (unsigned int)v58];
        if ( v57 != (_BYTE *)v33 )
        {
          do
          {
            v33 -= 24LL;
            if ( *(_DWORD *)(v33 + 16) > 0x40u )
            {
              v34 = *(_QWORD *)(v33 + 8);
              if ( v34 )
                j_j___libc_free_0_0(v34);
            }
          }
          while ( v32 != (_BYTE *)v33 );
LABEL_47:
          v33 = (unsigned __int64)v57;
          goto LABEL_48;
        }
        goto LABEL_48;
      }
    }
    j___libc_free_0(v74);
    if ( v69 <= 0x40 )
      goto LABEL_40;
    goto LABEL_38;
  }
  v35 = (__int64 *)a1[1];
  v36 = 3LL * *((unsigned int *)a1 + 4);
  v37 = (const __m128i *)&v35[v36];
  v38 = 0xAAAAAAAAAAAAAAABLL * ((v36 * 8) >> 3);
  if ( !(v38 >> 2) )
  {
    v39 = (__int64 *)a1[1];
LABEL_94:
    if ( v38 != 2 )
    {
      if ( v38 != 3 )
      {
        if ( v38 != 1 )
          goto LABEL_97;
        goto LABEL_107;
      }
      if ( (v39[2] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_63;
      v39 += 3;
    }
    if ( (v39[2] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_63;
    v39 += 3;
LABEL_107:
    if ( (v39[2] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_63;
LABEL_97:
    v39 = (__int64 *)v37;
LABEL_98:
    v42 = v39;
    goto LABEL_70;
  }
  v39 = (__int64 *)a1[1];
  while ( (v39[2] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v39[5] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v39 += 3;
      break;
    }
    if ( (v39[8] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v39 += 6;
      break;
    }
    if ( (v39[11] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v39 += 9;
      break;
    }
    v39 += 12;
    if ( &v35[12 * (v38 >> 2)] == v39 )
    {
      v38 = 0xAAAAAAAAAAAAAAABLL * (((char *)v37 - (char *)v39) >> 3);
      goto LABEL_94;
    }
  }
LABEL_63:
  if ( v37 == (const __m128i *)v39 )
    goto LABEL_98;
  v40 = (const __m128i *)(v39 + 3);
  if ( v37 == (const __m128i *)(v39 + 3) )
    goto LABEL_98;
  do
  {
    if ( (v40[1].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v39 += 3;
      *(__m128i *)(v39 - 3) = _mm_loadu_si128(v40);
      *(v39 - 1) = v40[1].m128i_i64[0];
    }
    v40 = (const __m128i *)((char *)v40 + 24);
  }
  while ( v37 != v40 );
  v35 = (__int64 *)a1[1];
  v41 = (char *)&v35[3 * *((unsigned int *)a1 + 4)] - (char *)v37;
  v42 = (__int64 *)((char *)v39 + v41);
  if ( v37 != (const __m128i *)&v35[3 * *((unsigned int *)a1 + 4)] )
  {
    memmove(v39, v37, v41);
    v35 = (__int64 *)a1[1];
  }
LABEL_70:
  v43 = -1431655765 * (v42 - v35);
  *((_DWORD *)a1 + 4) = v43;
  v44 = 3LL * v43;
  v45 = &v35[v44];
  if ( &v35[v44] != v35 )
  {
    _BitScanReverse64(&v46, 0xAAAAAAAAAAAAAAABLL * ((v44 * 8) >> 3));
    sub_1A1B940((__int64)v35, (__m128i *)&v35[v44], 2LL * (int)(63 - (v46 ^ 0x3F)), v12, v13, v14);
    if ( (unsigned __int64)v44 <= 48 )
    {
      sub_1A1AE20(v35, &v35[v44]);
    }
    else
    {
      v47 = (unsigned __int64 *)(v35 + 48);
      sub_1A1AE20(v35, v35 + 48);
      if ( v45 != v35 + 48 )
      {
        do
        {
          v48 = v47;
          v47 += 3;
          sub_1A1AC50(v48);
        }
        while ( v45 != (__int64 *)v47 );
      }
    }
  }
  if ( v80 != v79 )
    _libc_free((unsigned __int64)v80);
  if ( (v76 & 1) != 0 )
  {
    if ( (v73 & 1) != 0 )
      goto LABEL_78;
LABEL_89:
    j___libc_free_0(v74);
    if ( v69 <= 0x40 )
      goto LABEL_81;
    goto LABEL_79;
  }
  j___libc_free_0(v77);
  if ( (v73 & 1) == 0 )
    goto LABEL_89;
LABEL_78:
  if ( v69 <= 0x40 )
    goto LABEL_81;
LABEL_79:
  if ( v68 )
    j_j___libc_free_0_0(v68);
LABEL_81:
  if ( v62 != v61 )
    _libc_free((unsigned __int64)v62);
  v49 = v57;
  v33 = (unsigned __int64)&v57[24 * (unsigned int)v58];
  if ( v57 != (_BYTE *)v33 )
  {
    do
    {
      v33 -= 24LL;
      if ( *(_DWORD *)(v33 + 16) > 0x40u )
      {
        v50 = *(_QWORD *)(v33 + 8);
        if ( v50 )
          j_j___libc_free_0_0(v50);
      }
    }
    while ( v49 != (_BYTE *)v33 );
    goto LABEL_47;
  }
LABEL_48:
  if ( (_BYTE *)v33 != v59 )
    _libc_free(v33);
}
