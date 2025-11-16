// Function: sub_35DB0F0
// Address: 0x35db0f0
//
void __fastcall sub_35DB0F0(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 *v5; // r13
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int); // rax
  int v7; // eax
  unsigned __int16 v8; // dx
  __int64 v9; // rcx
  __int64 v10; // rax
  char v11; // cl
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // r12d
  int v15; // r14d
  signed __int64 v16; // rbx
  __int64 v17; // rbx
  char *v18; // rax
  char *v19; // rdx
  _DWORD *v20; // rax
  _BYTE *v21; // rdx
  _DWORD *i; // rbx
  signed __int64 v23; // rdx
  signed __int64 v24; // rbx
  __int64 v25; // r14
  __int64 *v26; // r12
  unsigned __int64 v27; // r15
  __int64 v28; // r9
  unsigned int v29; // r12d
  void *v30; // r8
  int v31; // eax
  unsigned __int64 v32; // r12
  __int64 v33; // r12
  unsigned int v34; // r15d
  __m128i *v35; // r15
  signed __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 *v39; // r15
  unsigned int v40; // ebx
  signed __int64 v41; // r12
  unsigned int v42; // r13d
  __int64 v43; // rax
  unsigned int v44; // ecx
  __int64 v45; // rdi
  __int64 v46; // rsi
  __int64 v47; // r13
  __int64 v48; // rsi
  __int64 v49; // rcx
  unsigned __int64 v50; // rdi
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // r12d
  __int64 *v55; // rbx
  __int64 *v56; // r14
  __int64 v57; // rdi
  unsigned int v58; // edx
  __int64 v59; // [rsp+10h] [rbp-150h]
  signed __int64 v61; // [rsp+20h] [rbp-140h]
  signed __int64 v62; // [rsp+28h] [rbp-138h]
  __int64 v63; // [rsp+30h] [rbp-130h]
  signed __int64 v64; // [rsp+38h] [rbp-128h]
  _QWORD *v65; // [rsp+48h] [rbp-118h]
  signed __int64 v66; // [rsp+48h] [rbp-118h]
  __int64 v67; // [rsp+50h] [rbp-110h]
  __int64 v69; // [rsp+68h] [rbp-F8h]
  unsigned int v70; // [rsp+68h] [rbp-F8h]
  void *v71; // [rsp+70h] [rbp-F0h] BYREF
  unsigned int v72; // [rsp+78h] [rbp-E8h]
  _DWORD *v73; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v74; // [rsp+88h] [rbp-D8h]
  _DWORD v75[8]; // [rsp+90h] [rbp-D0h] BYREF
  _BYTE *v76; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v77; // [rsp+B8h] [rbp-A8h]
  _BYTE v78[32]; // [rsp+C0h] [rbp-A0h] BYREF
  void *v79[2]; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v80; // [rsp+F0h] [rbp-70h] BYREF
  int v81; // [rsp+100h] [rbp-60h]
  int v82; // [rsp+120h] [rbp-40h]

  if ( !*(_DWORD *)(a1[11] + 648LL) )
    return;
  v4 = a1[10];
  v5 = a2;
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v4 + 32LL);
  if ( v6 == sub_2D42F30 )
  {
    v7 = sub_AE2980(a1[12], 0)[1];
    switch ( v7 )
    {
      case 1:
        v8 = 2;
        v9 = a1[10];
        break;
      case 2:
        v8 = 3;
        v9 = a1[10];
        break;
      case 4:
        v8 = 4;
        v9 = a1[10];
        break;
      case 8:
        v8 = 5;
        v9 = a1[10];
        break;
      case 16:
        v8 = 6;
        v9 = a1[10];
        break;
      case 32:
        v8 = 7;
        v9 = a1[10];
        break;
      case 64:
        v8 = 8;
        v9 = a1[10];
        break;
      case 128:
        v8 = 9;
        v9 = a1[10];
        break;
      default:
        return;
    }
  }
  else
  {
    v8 = v6(v4, a1[12], 0);
    v9 = a1[10];
    if ( v8 == 1 )
    {
      if ( *(_BYTE *)(v9 + 7104) )
        return;
      goto LABEL_106;
    }
    if ( !v8 )
      return;
  }
  if ( !*(_QWORD *)(v9 + 8LL * v8 + 112) || *(_BYTE *)(v9 + 500LL * v8 + 6604) )
    return;
  if ( (unsigned __int16)(v8 - 504) <= 7u )
LABEL_106:
    BUG();
  v10 = 16LL * (v8 - 1);
  v11 = byte_444C4A0[v10 + 8];
  v79[0] = *(void **)&byte_444C4A0[v10];
  LOBYTE(v79[1]) = v11;
  v74 = 0x800000000LL;
  v14 = sub_CA1930(v79);
  v59 = a2[1] - *a2;
  v62 = 0xCCCCCCCCCCCCCCCDLL * (v59 >> 3);
  v15 = v62 - 1;
  v73 = v75;
  v64 = v62 - 1;
  v16 = v62 - 1;
  if ( !v62 )
  {
    v77 = 0x800000000LL;
    v76 = v78;
    v75[v16] = 1;
    *(_DWORD *)&v76[4 * v64] = v15;
    goto LABEL_60;
  }
  v17 = v16 * 4 + 4;
  if ( (unsigned __int64)v59 > 0x140 )
  {
    sub_C8D5F0((__int64)&v73, v75, 0xCCCCCCCCCCCCCCCDLL * (v59 >> 3), 4u, v12, v13);
    v18 = (char *)&v73[(unsigned int)v74];
    v19 = (char *)v73 + v17;
    if ( v18 == (char *)v73 + v17 )
    {
      v77 = 0x800000000LL;
      LODWORD(v74) = -858993459 * (v59 >> 3);
      v76 = v78;
      goto LABEL_99;
    }
    do
    {
LABEL_21:
      if ( v18 )
        *(_DWORD *)v18 = 0;
      v18 += 4;
    }
    while ( v18 != v19 );
    v77 = 0x800000000LL;
    LODWORD(v74) = -858993459 * (v59 >> 3);
    v20 = v78;
    v76 = v78;
    if ( (unsigned __int64)v59 <= 0x140 )
      goto LABEL_25;
LABEL_99:
    sub_C8D5F0((__int64)&v76, v78, v62, 4u, v12, v13);
    v21 = v76;
    v20 = &v76[4 * (unsigned int)v77];
    goto LABEL_26;
  }
  v18 = (char *)v75;
  v19 = (char *)v75 + v17;
  if ( (_DWORD *)((char *)v75 + v17) != v75 )
    goto LABEL_21;
  v77 = 0x800000000LL;
  LODWORD(v74) = -858993459 * (v59 >> 3);
  v20 = v78;
  v76 = v78;
LABEL_25:
  v21 = v78;
LABEL_26:
  for ( i = &v21[v17]; i != v20; ++v20 )
  {
    if ( v20 )
      *v20 = 0;
  }
  LODWORD(v77) = -858993459 * (v59 >> 3);
  v73[v64] = 1;
  v69 = v62 - 2;
  *(_DWORD *)&v76[4 * v64] = v15;
  if ( v62 - 2 < 0 )
    goto LABEL_60;
  v23 = v69 + v14;
  v67 = v59 - 80;
  do
  {
    v63 = v69;
    v73[v69] = v73[v69 + 1] + 1;
    *(_DWORD *)&v76[4 * v69] = v69;
    v24 = v23 - 1;
    if ( v62 <= v23 )
      v24 = v62 - 1;
    v61 = v23 - 1;
    if ( v24 <= v69 )
      goto LABEL_59;
    while ( 1 )
    {
      v25 = *(_QWORD *)(*v5 + 40 * v24 + 16);
      v26 = (__int64 *)(*(_QWORD *)(*v5 + v67 + 8) + 24LL);
      v27 = (unsigned int)sub_AE2980(a1[12], 0)[3];
      LODWORD(v79[1]) = *(_DWORD *)(v25 + 32);
      if ( LODWORD(v79[1]) > 0x40 )
        sub_C43780((__int64)v79, (const void **)(v25 + 24));
      else
        v79[0] = *(void **)(v25 + 24);
      sub_C46B40((__int64)v79, v26);
      v29 = (unsigned int)v79[1];
      v30 = v79[0];
      LODWORD(v79[1]) = 0;
      v72 = v29;
      v71 = v79[0];
      if ( v29 > 0x40 )
      {
        v65 = v79[0];
        v31 = sub_C444A0((__int64)&v71);
        v30 = v65;
        if ( v29 - v31 <= 0x40 )
        {
          v32 = *v65;
          if ( *v65 == -1 )
          {
            j_j___libc_free_0_0((unsigned __int64)v65);
            if ( LODWORD(v79[1]) <= 0x40 )
              goto LABEL_37;
LABEL_45:
            if ( v79[0] )
              j_j___libc_free_0_0((unsigned __int64)v79[0]);
            goto LABEL_47;
          }
          ++v32;
        }
        else
        {
          v32 = -1;
        }
        if ( !v65 )
          goto LABEL_47;
        j_j___libc_free_0_0((unsigned __int64)v65);
        if ( LODWORD(v79[1]) <= 0x40 )
          goto LABEL_47;
        goto LABEL_45;
      }
      if ( v79[0] == (void *)-1LL )
        goto LABEL_37;
      v32 = (unsigned __int64)v79[0] + 1;
LABEL_47:
      if ( v32 > v27 )
        goto LABEL_37;
      v33 = (__int64)(*(_QWORD *)(*(_QWORD *)(a1[13] + 8LL) + 104LL) - *(_QWORD *)(*(_QWORD *)(a1[13] + 8LL) + 96LL)) >> 3;
      v79[0] = &v80;
      v34 = (unsigned int)(v33 + 63) >> 6;
      v79[1] = (void *)0x600000000LL;
      if ( v34 > 6 )
      {
        sub_C8D5F0((__int64)v79, &v80, v34, 8u, (__int64)v30, v28);
        memset(v79[0], 0, 8LL * v34);
        LODWORD(v79[1]) = (unsigned int)(v33 + 63) >> 6;
        v35 = (__m128i *)v79[0];
      }
      else
      {
        if ( v34 && 8LL * v34 )
          memset(&v80, 0, 8LL * v34);
        LODWORD(v79[1]) = (unsigned int)(v33 + 63) >> 6;
        v35 = &v80;
      }
      v36 = v69;
      v82 = v33;
      if ( v24 >= v69 )
        break;
LABEL_83:
      if ( (__m128i *)((char *)v35 + 8 * LODWORD(v79[1])) != v35 )
      {
        v66 = v24;
        v54 = 0;
        v55 = (__int64 *)v35;
        v56 = &v35->m128i_i64[LODWORD(v79[1])];
        do
        {
          v57 = *v55++;
          v54 += sub_39FAC40(v57);
        }
        while ( v55 != v56 );
        v24 = v66;
        if ( v54 > 3 )
          goto LABEL_57;
      }
      v58 = 1;
      if ( v24 != v64 )
        v58 = v73[v24 + 1] + 1;
      if ( v73[v63] > v58 )
      {
        v73[v63] = v58;
        *(_DWORD *)&v76[4 * v69] = v24;
        v35 = (__m128i *)v79[0];
      }
      if ( v35 != &v80 )
        _libc_free((unsigned __int64)v35);
LABEL_37:
      if ( --v24 == v69 )
        goto LABEL_59;
    }
    v37 = v67;
    while ( 1 )
    {
      v38 = v37 + *v5;
      if ( *(_DWORD *)v38 )
        break;
      ++v36;
      v37 += 40;
      v35->m128i_i64[*(_DWORD *)(*(_QWORD *)(v38 + 24) + 24LL) >> 6] |= 1LL << *(_DWORD *)(*(_QWORD *)(v38 + 24) + 24LL);
      v35 = (__m128i *)v79[0];
      if ( v36 > v24 )
        goto LABEL_83;
    }
LABEL_57:
    if ( v35 != &v80 )
      _libc_free((unsigned __int64)v35);
LABEL_59:
    --v69;
    v67 -= 40;
    v23 = v61;
  }
  while ( v69 != -1 );
LABEL_60:
  if ( v59 <= 0 )
  {
    v48 = v5[1];
    v49 = *v5;
    v50 = 0;
    v52 = 0xCCCCCCCCCCCCCCCDLL * ((v48 - *v5) >> 3);
    goto LABEL_66;
  }
  v39 = v5;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  do
  {
    while ( 1 )
    {
      v44 = *(_DWORD *)&v76[4 * v41];
      v81 = -1;
      v70 = v44;
      v45 = 5LL * v40;
      v46 = 40LL * v40;
      if ( !(unsigned __int8)sub_35DB0D0(a1, v39, v42, v44, a3, (__int64)v79) )
        break;
      v43 = *v39;
      v41 = v70 + 1;
      ++v40;
      v42 = v70 + 1;
      *(__m128i *)(v43 + 8 * v45) = _mm_loadu_si128((const __m128i *)v79);
      *(__m128i *)(v43 + v46 + 16) = _mm_loadu_si128(&v80);
      *(_DWORD *)(v43 + v46 + 32) = v81;
      if ( v41 >= v62 )
        goto LABEL_65;
    }
    v47 = 1 - v42 + v70;
    v40 += v47;
    memmove((void *)(*v39 + v46), (const void *)(*v39 + 40 * v41), 40 * v47);
    v41 = v70 + 1;
    v42 = v70 + 1;
  }
  while ( v41 < v62 );
LABEL_65:
  v48 = v39[1];
  v49 = *v39;
  v50 = v40;
  v5 = v39;
  v51 = 0xCCCCCCCCCCCCCCCDLL * ((v48 - *v39) >> 3);
  v52 = v51;
  if ( v51 < v40 )
  {
    sub_35D8D40((const __m128i **)v39, v40 - v51);
  }
  else
  {
LABEL_66:
    if ( v52 > v50 )
    {
      v53 = v49 + 40 * v50;
      if ( v48 != v53 )
        v5[1] = v53;
    }
  }
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
}
