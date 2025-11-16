// Function: sub_24D57A0
// Address: 0x24d57a0
//
__m128i *__fastcall sub_24D57A0(__m128i *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v4; // al
  int v5; // ecx
  __int64 v6; // r15
  __int64 i; // rdx
  _BYTE *v8; // r14
  char v10; // di
  __int64 v11; // rcx
  int v12; // eax
  unsigned int v13; // edx
  _QWORD *v14; // rsi
  _BYTE *v15; // r8
  __int64 v16; // rax
  unsigned __int8 v17; // al
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  char *v21; // rsi
  __int64 v22; // rdx
  unsigned __int8 v23; // al
  _BYTE **v24; // rax
  _BYTE *v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rcx
  _BYTE *v28; // rsi
  _BYTE *v29; // rdi
  int *v30; // rax
  __int64 v31; // rdi
  char v32; // cl
  __int64 v33; // r8
  int v34; // esi
  __int64 v35; // rax
  _QWORD *v36; // rdi
  _BYTE *v37; // r9
  unsigned __int64 *v38; // rdi
  unsigned __int64 v39; // rax
  unsigned int v40; // esi
  size_t v41; // rdx
  __int64 v42; // rsi
  int *v43; // rdi
  __int64 v44; // rsi
  int v45; // esi
  unsigned int v46; // eax
  _QWORD *v47; // rdx
  int v48; // edi
  unsigned int v49; // r8d
  __m128i *v50; // rax
  __int64 v51; // rcx
  size_t v52; // rdx
  int v53; // r11d
  int v54; // ecx
  __int64 v55; // r8
  int v56; // ecx
  __int64 v57; // rax
  _BYTE *v58; // r10
  int v59; // ecx
  __int64 v60; // r8
  int v61; // ecx
  __int64 v62; // rax
  __int64 v63; // r10
  int v64; // edi
  _QWORD *v65; // rsi
  int v66; // r9d
  int v67; // edi
  __int64 v69; // [rsp+18h] [rbp-178h]
  __int64 v70; // [rsp+20h] [rbp-170h]
  _DWORD v72[5]; // [rsp+40h] [rbp-150h] BYREF
  char v73; // [rsp+54h] [rbp-13Ch] BYREF
  _BYTE v74[11]; // [rsp+55h] [rbp-13Bh] BYREF
  void *dest; // [rsp+60h] [rbp-130h] BYREF
  size_t v76; // [rsp+68h] [rbp-128h]
  _QWORD v77[2]; // [rsp+70h] [rbp-120h] BYREF
  int *v78; // [rsp+80h] [rbp-110h] BYREF
  size_t n; // [rsp+88h] [rbp-108h]
  __int64 v80; // [rsp+90h] [rbp-100h] BYREF
  char v81; // [rsp+98h] [rbp-F8h] BYREF
  int v82[52]; // [rsp+C0h] [rbp-D0h] BYREF

  sub_C7D030(v82);
  v4 = *(_BYTE *)(a2 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_DWORD *)(a2 - 24);
  else
    v5 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  if ( v5 <= 1 )
  {
LABEL_87:
    sub_C7D290(v82, v72);
    sub_C7D470(&v78, (unsigned __int8 *)v72);
    dest = v77;
    sub_24D2A10((__int64 *)&dest, v78, (__int64)v78 + n);
    v50 = (__m128i *)sub_2241130((unsigned __int64 *)&dest, 0, 0, "__anonymous_", 0xCu);
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( (__m128i *)v50->m128i_i64[0] == &v50[1] )
    {
      a1[1] = _mm_loadu_si128(v50 + 1);
    }
    else
    {
      a1->m128i_i64[0] = v50->m128i_i64[0];
      a1[1].m128i_i64[0] = v50[1].m128i_i64[0];
    }
    v51 = v50->m128i_i64[1];
    v50->m128i_i64[0] = (__int64)v50[1].m128i_i64;
    v50->m128i_i64[1] = 0;
    a1->m128i_i64[1] = v51;
    v50[1].m128i_i8[0] = 0;
    if ( dest != v77 )
      j_j___libc_free_0((unsigned __int64)dest);
    if ( v78 != (int *)&v81 )
      _libc_free((unsigned __int64)v78);
    return a1;
  }
  v6 = 16;
  v70 = a2 - 16;
  v69 = 16LL * (((unsigned int)(v5 - 2) >> 1) + 2);
  if ( (v4 & 2) == 0 )
    goto LABEL_28;
LABEL_5:
  for ( i = *(_QWORD *)(a2 - 32); ; i = v70 - 8LL * ((v4 >> 2) & 0xF) )
  {
    v8 = *(_BYTE **)(i + v6 - 8);
    if ( (unsigned __int8)(*v8 - 5) > 0x1Fu )
    {
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      sub_24D2A10(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
      return a1;
    }
    v10 = *(_BYTE *)(a3 + 8) & 1;
    if ( v10 )
    {
      v11 = a3 + 16;
      v12 = 7;
    }
    else
    {
      v22 = *(unsigned int *)(a3 + 24);
      v11 = *(_QWORD *)(a3 + 16);
      if ( !(_DWORD)v22 )
        goto LABEL_61;
      v12 = v22 - 1;
    }
    v13 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v14 = (_QWORD *)(v11 + 40LL * v13);
    v15 = (_BYTE *)*v14;
    if ( v8 == (_BYTE *)*v14 )
      goto LABEL_12;
    v45 = 1;
    while ( v15 != (_BYTE *)-4096LL )
    {
      v66 = v45 + 1;
      v13 = v12 & (v45 + v13);
      v14 = (_QWORD *)(v11 + 40LL * v13);
      v15 = (_BYTE *)*v14;
      if ( v8 == (_BYTE *)*v14 )
        goto LABEL_12;
      v45 = v66;
    }
    if ( v10 )
    {
      v42 = 320;
      goto LABEL_62;
    }
    v22 = *(unsigned int *)(a3 + 24);
LABEL_61:
    v42 = 40 * v22;
LABEL_62:
    v14 = (_QWORD *)(v11 + v42);
LABEL_12:
    LOBYTE(v77[0]) = 0;
    dest = v77;
    v16 = 320;
    v76 = 0;
    if ( !v10 )
      v16 = 40LL * *(unsigned int *)(a3 + 24);
    if ( v14 != (_QWORD *)(v11 + v16) )
    {
      sub_2240AE0((unsigned __int64 *)&dest, v14 + 1);
      goto LABEL_16;
    }
    v23 = *(v8 - 16);
    if ( (v23 & 2) != 0 )
    {
      if ( !*((_DWORD *)v8 - 6) )
        goto LABEL_104;
      v24 = (_BYTE **)*((_QWORD *)v8 - 4);
      v25 = *v24;
      if ( **v24 )
      {
LABEL_55:
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        sub_24D2A10(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
        goto LABEL_56;
      }
    }
    else
    {
      if ( (*((_WORD *)v8 - 8) & 0x3C0) == 0 )
      {
LABEL_104:
        a1->m128i_i64[1] = 0;
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        a1[1].m128i_i8[0] = 0;
        return a1;
      }
      v25 = *(_BYTE **)&v8[-8 * ((v23 >> 2) & 0xF) - 16];
      if ( *v25 )
        goto LABEL_55;
    }
    v28 = (_BYTE *)sub_B91420((__int64)v25);
    if ( !v28 )
    {
      LOBYTE(v80) = 0;
      v41 = 0;
      v29 = dest;
      v78 = (int *)&v80;
LABEL_59:
      v76 = v41;
      v29[v41] = 0;
      v30 = v78;
      goto LABEL_39;
    }
    v78 = (int *)&v80;
    sub_24D2A10((__int64 *)&v78, v28, (__int64)&v28[v26]);
    v29 = dest;
    v30 = (int *)dest;
    if ( v78 == (int *)&v80 )
    {
      v41 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = v80;
        else
          memcpy(dest, &v80, n);
        v41 = n;
        v29 = dest;
      }
      goto LABEL_59;
    }
    v27 = v80;
    if ( dest == v77 )
    {
      dest = v78;
      v76 = n;
      v77[0] = v80;
    }
    else
    {
      v31 = v77[0];
      dest = v78;
      v76 = n;
      v77[0] = v80;
      if ( v30 )
      {
        v78 = v30;
        v80 = v31;
        goto LABEL_39;
      }
    }
    v30 = (int *)&v80;
    v78 = (int *)&v80;
LABEL_39:
    n = 0;
    *(_BYTE *)v30 = 0;
    if ( v78 != (int *)&v80 )
      j_j___libc_free_0((unsigned __int64)v78);
    if ( !v76 )
      break;
LABEL_42:
    v32 = *(_BYTE *)(a3 + 8) & 1;
    if ( v32 )
    {
      v33 = a3 + 16;
      v34 = 7;
    }
    else
    {
      v40 = *(_DWORD *)(a3 + 24);
      v33 = *(_QWORD *)(a3 + 16);
      if ( !v40 )
      {
        v46 = *(_DWORD *)(a3 + 8);
        ++*(_QWORD *)a3;
        v47 = 0;
        v48 = (v46 >> 1) + 1;
        goto LABEL_77;
      }
      v34 = v40 - 1;
    }
    LODWORD(v35) = v34 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v36 = (_QWORD *)(v33 + 40LL * (unsigned int)v35);
    v37 = (_BYTE *)*v36;
    if ( v8 != (_BYTE *)*v36 )
    {
      v53 = 1;
      v47 = 0;
      while ( v37 != (_BYTE *)-4096LL )
      {
        if ( !v47 && v37 == (_BYTE *)-8192LL )
          v47 = v36;
        v35 = v34 & (unsigned int)(v35 + v53);
        v36 = (_QWORD *)(v33 + 40 * v35);
        v37 = (_BYTE *)*v36;
        if ( v8 == (_BYTE *)*v36 )
          goto LABEL_45;
        ++v53;
      }
      v46 = *(_DWORD *)(a3 + 8);
      v49 = 24;
      v40 = 8;
      if ( !v47 )
        v47 = v36;
      ++*(_QWORD *)a3;
      v48 = (v46 >> 1) + 1;
      if ( v32 )
      {
LABEL_78:
        if ( v49 <= 4 * v48 )
        {
          sub_24D5590(a3, 2 * v40);
          if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
          {
            v55 = a3 + 16;
            v56 = 7;
          }
          else
          {
            v54 = *(_DWORD *)(a3 + 24);
            v55 = *(_QWORD *)(a3 + 16);
            if ( !v54 )
              goto LABEL_146;
            v56 = v54 - 1;
          }
          LODWORD(v57) = v56 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v47 = (_QWORD *)(v55 + 40LL * (unsigned int)v57);
          v58 = (_BYTE *)*v47;
          if ( v8 == (_BYTE *)*v47 )
            goto LABEL_110;
          v67 = 1;
          v65 = 0;
          while ( v58 != (_BYTE *)-4096LL )
          {
            if ( !v65 && v58 == (_BYTE *)-8192LL )
              v65 = v47;
            v57 = v56 & (unsigned int)(v57 + v67);
            v47 = (_QWORD *)(v55 + 40 * v57);
            v58 = (_BYTE *)*v47;
            if ( v8 == (_BYTE *)*v47 )
              goto LABEL_110;
            ++v67;
          }
        }
        else
        {
          if ( v40 - *(_DWORD *)(a3 + 12) - v48 > v40 >> 3 )
          {
LABEL_80:
            *(_DWORD *)(a3 + 8) = (2 * (v46 >> 1) + 2) | v46 & 1;
            if ( *v47 != -4096 )
              --*(_DWORD *)(a3 + 12);
            *v47 = v8;
            v38 = v47 + 1;
            v47[1] = v47 + 3;
            v47[2] = 0;
            *((_BYTE *)v47 + 24) = 0;
            goto LABEL_46;
          }
          sub_24D5590(a3, v40);
          if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
          {
            v60 = a3 + 16;
            v61 = 7;
          }
          else
          {
            v59 = *(_DWORD *)(a3 + 24);
            v60 = *(_QWORD *)(a3 + 16);
            if ( !v59 )
            {
LABEL_146:
              *(_DWORD *)(a3 + 8) = (2 * (*(_DWORD *)(a3 + 8) >> 1) + 2) | *(_DWORD *)(a3 + 8) & 1;
              BUG();
            }
            v61 = v59 - 1;
          }
          LODWORD(v62) = v61 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v47 = (_QWORD *)(v60 + 40LL * (unsigned int)v62);
          v63 = *v47;
          if ( v8 == (_BYTE *)*v47 )
          {
LABEL_110:
            v46 = *(_DWORD *)(a3 + 8);
            goto LABEL_80;
          }
          v64 = 1;
          v65 = 0;
          while ( v63 != -4096 )
          {
            if ( v63 == -8192 && !v65 )
              v65 = v47;
            v62 = v61 & (unsigned int)(v62 + v64);
            v47 = (_QWORD *)(v60 + 40 * v62);
            v63 = *v47;
            if ( v8 == (_BYTE *)*v47 )
              goto LABEL_110;
            ++v64;
          }
        }
        if ( v65 )
          v47 = v65;
        goto LABEL_110;
      }
      v40 = *(_DWORD *)(a3 + 24);
LABEL_77:
      v49 = 3 * v40;
      goto LABEL_78;
    }
LABEL_45:
    v38 = v36 + 1;
LABEL_46:
    sub_2240AE0(v38, (unsigned __int64 *)&dest);
LABEL_16:
    sub_C7D280(v82, (int *)dest, v76);
    sub_C7D280(v82, dword_438817C, 0);
    v17 = *(_BYTE *)(a2 - 16);
    if ( (v17 & 2) != 0 )
      v18 = *(_QWORD *)(a2 - 32);
    else
      v18 = v70 - 8LL * ((v17 >> 2) & 0xF);
    v19 = *(_QWORD *)(*(_QWORD *)(v18 + v6) + 136LL);
    v20 = *(_QWORD *)(v19 + 24);
    if ( *(_DWORD *)(v19 + 32) > 0x40u )
      v20 = *(_QWORD *)v20;
    if ( v20 )
    {
      v21 = v74;
      do
      {
        *--v21 = v20 % 0xA + 48;
        v39 = v20;
        v20 /= 0xAu;
      }
      while ( v39 > 9 );
    }
    else
    {
      v73 = 48;
      v21 = &v73;
    }
    v78 = (int *)&v80;
    sub_24D2960((__int64 *)&v78, v21, (__int64)v74);
    sub_C7D280(v82, v78, n);
    if ( v78 != (int *)&v80 )
      j_j___libc_free_0((unsigned __int64)v78);
    sub_C7D280(v82, dword_438817C, 0);
    if ( dest != v77 )
      j_j___libc_free_0((unsigned __int64)dest);
    v6 += 16;
    if ( v6 == v69 )
      goto LABEL_87;
    v4 = *(_BYTE *)(a2 - 16);
    if ( (v4 & 2) != 0 )
      goto LABEL_5;
LABEL_28:
    ;
  }
  sub_24D57A0(&v78, v8, a3, v27);
  v43 = (int *)dest;
  if ( v78 == (int *)&v80 )
  {
    v52 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = v80;
      else
        memcpy(dest, &v80, n);
      v52 = n;
      v43 = (int *)dest;
    }
    v76 = v52;
    *((_BYTE *)v43 + v52) = 0;
    v43 = v78;
  }
  else
  {
    if ( dest == v77 )
    {
      dest = v78;
      v76 = n;
      v77[0] = v80;
    }
    else
    {
      v44 = v77[0];
      dest = v78;
      v76 = n;
      v77[0] = v80;
      if ( v43 )
      {
        v78 = v43;
        v80 = v44;
        goto LABEL_67;
      }
    }
    v78 = (int *)&v80;
    v43 = (int *)&v80;
  }
LABEL_67:
  n = 0;
  *(_BYTE *)v43 = 0;
  if ( v78 != (int *)&v80 )
    j_j___libc_free_0((unsigned __int64)v78);
  if ( v76 )
    goto LABEL_42;
  a1->m128i_i64[1] = 0;
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  a1[1].m128i_i8[0] = 0;
LABEL_56:
  if ( dest != v77 )
    j_j___libc_free_0((unsigned __int64)dest);
  return a1;
}
