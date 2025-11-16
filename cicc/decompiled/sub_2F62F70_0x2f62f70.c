// Function: sub_2F62F70
// Address: 0x2f62f70
//
__int64 __fastcall sub_2F62F70(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v4; // rdx
  __int64 v5; // rsi
  __int64 v6; // r9
  unsigned __int64 v7; // rax
  __int64 v8; // r10
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // r13
  __int64 v20; // rbx
  unsigned __int64 v21; // r11
  _QWORD *v22; // r10
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // r9
  const __m128i *v26; // r15
  __m128i *v27; // rax
  __int64 v28; // rdx
  unsigned int v29; // ecx
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // rdx
  __int16 v37; // ax
  __int64 v38; // rbx
  int v39; // r14d
  _BYTE *v40; // r12
  __int64 v41; // rax
  _BYTE *v42; // r15
  _BYTE *v43; // rbx
  char v44; // al
  __int64 v45; // rdx
  _QWORD *v46; // rdi
  unsigned int v47; // eax
  _BYTE *v48; // r15
  _QWORD *v49; // rax
  char *v50; // r15
  __int64 v51; // rax
  _QWORD *v52; // rdi
  unsigned int v53; // ebx
  __int64 v54; // r8
  __int64 *v55; // rcx
  __int64 v56; // rcx
  _QWORD *v57; // rdi
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 *v60; // rsi
  __int64 v61; // [rsp+0h] [rbp-180h]
  __int64 v62; // [rsp+8h] [rbp-178h]
  __int64 v64; // [rsp+28h] [rbp-158h]
  unsigned int v65; // [rsp+30h] [rbp-150h]
  _QWORD *v66; // [rsp+30h] [rbp-150h]
  __int64 v67; // [rsp+38h] [rbp-148h]
  unsigned __int64 v68; // [rsp+38h] [rbp-148h]
  __int64 v70; // [rsp+48h] [rbp-138h]
  __int64 v71; // [rsp+50h] [rbp-130h]
  __int64 v72; // [rsp+58h] [rbp-128h]
  unsigned int v73; // [rsp+58h] [rbp-128h]
  unsigned int v74; // [rsp+58h] [rbp-128h]
  _QWORD v75[4]; // [rsp+60h] [rbp-120h] BYREF
  unsigned __int64 v76; // [rsp+80h] [rbp-100h] BYREF
  __int64 v77; // [rsp+88h] [rbp-F8h]
  _BYTE v78[240]; // [rsp+90h] [rbp-F0h] BYREF

  v64 = 0;
  v2 = *(unsigned int *)(*a1 + 72LL);
  v61 = 8 * v2;
  if ( !(_DWORD)v2 )
    return 1;
  while ( 1 )
  {
    v62 = a1[16] + 8 * v64;
    if ( *(_DWORD *)v62 == 4 )
      break;
LABEL_3:
    v64 += 8;
    if ( v61 == v64 )
      return 1;
  }
  if ( *((_BYTE *)a1 + 32) )
    return 0;
  v4 = *(_QWORD *)(*a1 + 64LL);
  v5 = a1[8];
  v6 = *(_QWORD *)(v4 + v64);
  v7 = *(_QWORD *)(a2 + 128) + ((unsigned __int64)**(unsigned int **)(v62 + 48) << 6);
  v70 = *(_QWORD *)(v62 + 16) & *(_QWORD *)(v7 + 32);
  v71 = *(_QWORD *)(v62 + 8) & *(_QWORD *)(v7 + 24);
  v76 = (unsigned __int64)v78;
  v77 = 0x800000000LL;
  v8 = *(_QWORD *)(v4 + v64);
  v9 = *(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v10 = *(_QWORD *)(v9 + 16);
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 24);
  }
  else
  {
    v51 = *(unsigned int *)(v5 + 304);
    v52 = *(_QWORD **)(v5 + 296);
    if ( *(_DWORD *)(v5 + 304) )
    {
      v53 = *(_DWORD *)(v9 + 24) | (*(__int64 *)(v8 + 8) >> 1) & 3;
      do
      {
        while ( 1 )
        {
          v54 = v51 >> 1;
          v55 = &v52[2 * (v51 >> 1)];
          if ( v53 < (*(_DWORD *)((*v55 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v55 >> 1) & 3) )
            break;
          v52 = v55 + 2;
          v51 = v51 - v54 - 1;
          if ( v51 <= 0 )
            goto LABEL_75;
        }
        v51 >>= 1;
      }
      while ( v54 > 0 );
    }
LABEL_75:
    v11 = *(v52 - 1);
  }
  v72 = v6;
  v12 = *(_QWORD *)(*(_QWORD *)(v5 + 152) + 16LL * *(unsigned int *)(v11 + 24) + 8);
  v13 = v12 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = sub_2E09D00(*(__int64 **)a2, *(_QWORD *)(v8 + 8));
  v14 = *(_QWORD *)(v15 + 8);
  v16 = (v12 >> 1) & 3;
  v17 = (v14 >> 1) & 3;
  if ( (*(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)v16) <= ((unsigned int)v17
                                                                              | *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                          + 24)) )
    goto LABEL_30;
  v18 = v70;
  v19 = v71;
  v20 = v72;
  v21 = v13;
  v22 = (_QWORD *)a2;
  while ( v17 != 3 )
  {
    v23 = (unsigned int)v77;
    v75[0] = v14;
    v75[1] = v19;
    v24 = v76;
    v25 = (unsigned int)v77 + 1LL;
    v75[2] = v18;
    v26 = (const __m128i *)v75;
    if ( v25 > HIDWORD(v77) )
    {
      if ( v76 > (unsigned __int64)v75 )
      {
        v66 = v22;
        v68 = v21;
        v74 = v16;
LABEL_60:
        sub_C8D5F0((__int64)&v76, v78, (unsigned int)v77 + 1LL, 0x18u, v16, v25);
        v24 = v76;
        v23 = (unsigned int)v77;
        v16 = v74;
        v21 = v68;
        v22 = v66;
        goto LABEL_12;
      }
      v66 = v22;
      v68 = v21;
      v74 = v16;
      if ( (unsigned __int64)v75 >= v76 + 24LL * (unsigned int)v77 )
        goto LABEL_60;
      v50 = (char *)v75 - v76;
      sub_C8D5F0((__int64)&v76, v78, (unsigned int)v77 + 1LL, 0x18u, v16, v25);
      v24 = v76;
      v23 = (unsigned int)v77;
      v22 = v66;
      v21 = v68;
      v16 = v74;
      v26 = (const __m128i *)&v50[v76];
    }
LABEL_12:
    v15 += 24;
    v27 = (__m128i *)(v24 + 24 * v23);
    *v27 = _mm_loadu_si128(v26);
    v28 = v26[1].m128i_i64[0];
    LODWORD(v77) = v77 + 1;
    v27[1].m128i_i64[0] = v28;
    if ( v15 == *(_QWORD *)*v22 + 24LL * *(unsigned int *)(*v22 + 8LL) )
      break;
    v29 = v16 | *(_DWORD *)(v21 + 24);
    if ( (*(_DWORD *)((*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v15 >> 1) & 3) >= v29 )
      break;
    v30 = (_QWORD *)(v22[16] + ((unsigned __int64)**(unsigned int **)(v15 + 16) << 6));
    v19 &= ~v30[1];
    v18 &= ~v30[2];
    if ( !v30[5] || !(v18 | v19) )
      break;
    v14 = *(_QWORD *)(v15 + 8);
    v17 = (v14 >> 1) & 3;
    if ( ((unsigned int)v17 | *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24)) >= v29 )
      goto LABEL_30;
  }
  v31 = *(_QWORD *)(v20 + 8);
  v32 = v31 >> 1;
  v33 = v31 & 0xFFFFFFFFFFFFFFF8LL;
  v34 = v32 & 3;
  v35 = *(_QWORD *)(v33 + 16);
  if ( v35 )
  {
    v36 = *(_QWORD *)(v35 + 24);
  }
  else
  {
    v56 = a1[8];
    v57 = *(_QWORD **)(v56 + 296);
    v58 = *(unsigned int *)(v56 + 304);
    if ( v58 )
    {
      do
      {
        while ( 1 )
        {
          v59 = v58 >> 1;
          v60 = &v57[2 * (v58 >> 1)];
          if ( ((unsigned int)v34 | *(_DWORD *)(v33 + 24)) < (*(_DWORD *)((*v60 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                            | (unsigned int)(*v60 >> 1) & 3) )
            break;
          v57 = v60 + 2;
          v58 = v58 - v59 - 1;
          if ( v58 <= 0 )
            goto LABEL_81;
        }
        v58 >>= 1;
      }
      while ( v59 > 0 );
    }
LABEL_81:
    v36 = *(v57 - 1);
  }
  if ( v34 )
  {
    if ( v34 != 1 )
    {
      if ( !v35 )
        BUG();
      if ( (*(_BYTE *)v35 & 4) != 0 )
      {
        v35 = *(_QWORD *)(v35 + 8);
      }
      else
      {
        while ( (*(_BYTE *)(v35 + 44) & 8) != 0 )
          v35 = *(_QWORD *)(v35 + 8);
        v35 = *(_QWORD *)(v35 + 8);
      }
    }
  }
  else
  {
    v35 = *(_QWORD *)(v36 + 56);
  }
  v65 = 0;
  v67 = *(_QWORD *)((*(_QWORD *)v76 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  while ( 1 )
  {
    v37 = *(_WORD *)(v35 + 68);
    if ( (unsigned __int16)(v37 - 14) > 4u && v37 != 24 )
    {
      v38 = *(_QWORD *)(v35 + 32);
      v39 = *(_DWORD *)(a2 + 8);
      v73 = *(_DWORD *)(a2 + 12);
      v40 = (_BYTE *)(v38 + 40LL * (*(_DWORD *)(v35 + 40) & 0xFFFFFF));
      v41 = 5LL * (unsigned int)sub_2E88FE0(v35);
      if ( v40 != (_BYTE *)(v38 + 8 * v41) )
      {
        v42 = (_BYTE *)(v38 + 8 * v41);
        while ( 1 )
        {
          v43 = v42;
          if ( (unsigned __int8)sub_2E2FA70(v42) )
            break;
          v42 += 40;
          if ( v40 == v42 )
            goto LABEL_24;
        }
        if ( v42 != v40 )
          break;
      }
    }
LABEL_24:
    if ( v67 == v35 )
    {
      if ( ++v65 == (_DWORD)v77 )
      {
        *(_DWORD *)v62 = 3;
        if ( (_BYTE *)v76 != v78 )
          _libc_free(v76);
        goto LABEL_3;
      }
      v49 = (_QWORD *)(v76 + 24LL * v65);
      v67 = *(_QWORD *)((*v49 & 0xFFFFFFFFFFFFFFF8LL) + 16);
      v71 = v49[1];
      v70 = v49[2];
    }
    if ( (*(_BYTE *)v35 & 4) == 0 && (*(_BYTE *)(v35 + 44) & 8) != 0 )
    {
      do
        v35 = *(_QWORD *)(v35 + 8);
      while ( (*(_BYTE *)(v35 + 44) & 8) != 0 );
    }
    v35 = *(_QWORD *)(v35 + 8);
  }
  while ( 1 )
  {
    if ( *((_DWORD *)v43 + 2) == v39 )
    {
      v44 = v43[4];
      if ( (v44 & 1) == 0 && (v44 & 2) == 0 )
      {
        LOWORD(v45) = (*(_DWORD *)v43 >> 8) & 0xFFF;
        if ( (v43[3] & 0x10) == 0 || (_WORD)v45 )
        {
          v45 = (unsigned __int16)v45;
          v46 = (_QWORD *)a1[9];
          if ( v73 )
          {
            if ( (_WORD)v45 )
            {
              v47 = (*(__int64 (__fastcall **)(_QWORD *))(*v46 + 296LL))(v46);
              v46 = (_QWORD *)a1[9];
              v45 = v47;
            }
            else
            {
              v45 = v73;
            }
          }
          if ( *(_QWORD *)(v46[34] + 16 * v45) & v71 | *(_QWORD *)(v46[34] + 16 * v45 + 8) & v70 )
            break;
        }
      }
    }
    if ( v43 + 40 != v40 )
    {
      v48 = v43 + 40;
      while ( 1 )
      {
        v43 = v48;
        if ( (unsigned __int8)sub_2E2FA70(v48) )
          break;
        v48 += 40;
        if ( v40 == v48 )
          goto LABEL_24;
      }
      if ( v40 != v48 )
        continue;
    }
    goto LABEL_24;
  }
LABEL_30:
  if ( (_BYTE *)v76 != v78 )
    _libc_free(v76);
  return 0;
}
