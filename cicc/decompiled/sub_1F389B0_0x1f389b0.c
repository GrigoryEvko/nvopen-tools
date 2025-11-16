// Function: sub_1F389B0
// Address: 0x1f389b0
//
__int64 __fastcall sub_1F389B0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned __int64 *v6; // rax
  __int64 *v9; // rsi
  __int64 *v10; // r8
  __int64 v12; // rsi
  __int64 v13; // r8
  unsigned __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r13
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned int v22; // edx
  unsigned int *v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  unsigned int *v27; // r15
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rbx
  __int64 i; // rdi
  unsigned __int16 v33; // ax
  int v34; // r14d
  unsigned int v35; // edx
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rdi
  __int64 v40; // r14
  __int64 v41; // rbx
  __int64 v42; // r13
  __int64 v43; // r12
  __int64 v44; // rax
  int v45; // r15d
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // rax
  int v50; // ebx
  _BYTE *v51; // rcx
  size_t v52; // r13
  _BYTE *v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rdi
  int v56; // edx
  int v57; // ebx
  unsigned int v58; // eax
  _DWORD *v59; // rdi
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rax
  __int64 v62; // rax
  _DWORD *v63; // rax
  __int64 v64; // rdx
  _DWORD *j; // rdx
  _DWORD *v66; // rax
  unsigned __int8 v67; // [rsp+1Fh] [rbp-261h]
  __int64 v68; // [rsp+28h] [rbp-258h]
  __int64 v71; // [rsp+38h] [rbp-248h]
  __int64 v73; // [rsp+40h] [rbp-240h]
  int v75; // [rsp+48h] [rbp-238h]
  _BYTE v76[48]; // [rsp+50h] [rbp-230h] BYREF
  _BYTE *v77; // [rsp+80h] [rbp-200h] BYREF
  __int64 v78; // [rsp+88h] [rbp-1F8h]
  _BYTE v79[64]; // [rsp+90h] [rbp-1F0h] BYREF
  unsigned __int64 v80[2]; // [rsp+D0h] [rbp-1B0h] BYREF
  _BYTE v81[64]; // [rsp+E0h] [rbp-1A0h] BYREF
  _BYTE *v82; // [rsp+120h] [rbp-160h] BYREF
  __int64 v83; // [rsp+128h] [rbp-158h]
  _BYTE v84[128]; // [rsp+130h] [rbp-150h] BYREF
  __int64 v85; // [rsp+1B0h] [rbp-D0h] BYREF
  __int64 v86; // [rsp+1B8h] [rbp-C8h]
  __int64 v87; // [rsp+1C0h] [rbp-C0h] BYREF
  unsigned __int64 v88[2]; // [rsp+200h] [rbp-80h] BYREF
  _BYTE v89[112]; // [rsp+210h] [rbp-70h] BYREF

  v6 = (unsigned __int64 *)&v87;
  v9 = *(__int64 **)(a3 + 88);
  v85 = 0;
  v86 = 1;
  v10 = *(__int64 **)(a3 + 96);
  do
    *v6++ = -8;
  while ( v6 != v88 );
  v88[0] = (unsigned __int64)v89;
  v88[1] = 0x800000000LL;
  sub_1F36F80((__int64)&v85, v9, v10, a4, (__int64)v10, a6);
  v77 = v79;
  v82 = v84;
  v78 = 0x800000000LL;
  v83 = 0x1000000000LL;
  v67 = sub_1F372D0((__int64 *)a1, a2, a3, a4, (__int64)&v77, (__int64)&v82);
  if ( !v67 )
    goto LABEL_4;
  v80[1] = 0x800000000LL;
  v12 = *(_QWORD *)(a1 + 40);
  v80[0] = (unsigned __int64)v81;
  sub_21073E0(v76, v12, v80);
  if ( *(_QWORD *)(a3 + 64) != *(_QWORD *)(a3 + 72) || *(_BYTE *)(a3 + 181) )
  {
    if ( *(_BYTE *)(a1 + 48) )
      sub_1F33C80(a1, a3, 0, (__int64)&v77, (__int64)&v85, v14);
  }
  else
  {
    if ( *(_BYTE *)(a1 + 48) )
      sub_1F33C80(a1, a3, 1, (__int64)&v77, (__int64)&v85, v14);
    sub_1F353B0(a1, (_QWORD *)a3, (__int64)a6);
  }
  v15 = *(unsigned int *)(a1 + 64);
  if ( !(_DWORD)v15 )
    goto LABEL_56;
  v73 = 0;
  v68 = 4 * v15;
  do
  {
    v16 = *(unsigned int *)(*(_QWORD *)(a1 + 56) + v73);
    sub_2107470(v76, v16);
    v17 = sub_1E69D00(*(_QWORD *)(a1 + 32), v16);
    v19 = v17;
    if ( v17 )
    {
      v19 = *(_QWORD *)(v17 + 24);
      sub_2107970(v76, v19, (unsigned int)v16);
    }
    v20 = *(_QWORD *)(a1 + 144);
    v21 = *(unsigned int *)(a1 + 160);
    if ( (_DWORD)v21 )
    {
      v13 = 1;
      v22 = (v21 - 1) & (37 * v16);
      v23 = (unsigned int *)(v20 + 32LL * v22);
      v18 = *v23;
      if ( (_DWORD)v16 == (_DWORD)v18 )
        goto LABEL_22;
      while ( (_DWORD)v18 != -1 )
      {
        LODWORD(v14) = v13 + 1;
        v22 = (v21 - 1) & (v13 + v22);
        v23 = (unsigned int *)(v20 + 32LL * v22);
        v18 = *v23;
        if ( (_DWORD)v16 == (_DWORD)v18 )
          goto LABEL_22;
        v13 = (unsigned int)v14;
      }
    }
    v23 = (unsigned int *)(v20 + 32 * v21);
LABEL_22:
    v24 = *((_QWORD *)v23 + 1);
    v25 = (*((_QWORD *)v23 + 2) - v24) >> 4;
    if ( (_DWORD)v25 )
    {
      v71 = v19;
      v26 = 0;
      v27 = v23;
      v28 = 16LL * (unsigned int)(v25 - 1);
      while ( 1 )
      {
        sub_2107970(v76, *(_QWORD *)(v26 + v24), *(unsigned int *)(v26 + v24 + 8));
        if ( v26 == v28 )
          break;
        v24 = *((_QWORD *)v27 + 1);
        v26 += 16;
      }
      v19 = v71;
    }
    v29 = *(_QWORD *)(a1 + 32);
    if ( (int)v16 < 0 )
      v30 = *(_QWORD *)(*(_QWORD *)(v29 + 24) + 16 * (v16 & 0x7FFFFFFF) + 8);
    else
      v30 = *(_QWORD *)(*(_QWORD *)(v29 + 272) + 8 * v16);
    if ( v30 )
    {
      if ( (*(_BYTE *)(v30 + 3) & 0x10) != 0 )
      {
        while ( 1 )
        {
          v30 = *(_QWORD *)(v30 + 32);
          if ( !v30 )
            break;
          if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
            goto LABEL_31;
        }
      }
      else
      {
        while ( 1 )
        {
LABEL_31:
          v31 = *(_QWORD *)(v30 + 32);
          for ( i = *(_QWORD *)(v30 + 16); v31; v31 = *(_QWORD *)(v31 + 32) )
          {
            if ( (*(_BYTE *)(v31 + 3) & 0x10) == 0 )
              break;
          }
          v33 = **(_WORD **)(i + 16);
          if ( v33 == 12 )
          {
            sub_1E16240(i);
          }
          else if ( v19 != *(_QWORD *)(i + 24) || (v29 = v33, v33 == 45) || !v33 )
          {
            sub_210B7D0(v76, v30, v29, v18, v13);
          }
          if ( !v31 )
            break;
          v30 = v31;
        }
      }
    }
    v73 += 4;
  }
  while ( v68 != v73 );
  v34 = *(_DWORD *)(a1 + 152);
  ++*(_QWORD *)(a1 + 136);
  *(_DWORD *)(a1 + 64) = 0;
  if ( v34 || *(_DWORD *)(a1 + 156) )
  {
    v35 = 4 * v34;
    v36 = *(_QWORD *)(a1 + 144);
    v37 = *(unsigned int *)(a1 + 160);
    v38 = v36 + 32 * v37;
    if ( (unsigned int)(4 * v34) < 0x40 )
      v35 = 64;
    if ( v35 < (unsigned int)v37 )
    {
      do
      {
        if ( *(_DWORD *)v36 <= 0xFFFFFFFD )
        {
          v55 = *(_QWORD *)(v36 + 8);
          if ( v55 )
            j_j___libc_free_0(v55, *(_QWORD *)(v36 + 24) - v55);
        }
        v36 += 32;
      }
      while ( v36 != v38 );
      v56 = *(_DWORD *)(a1 + 160);
      if ( v34 )
      {
        v57 = 64;
        if ( v34 != 1 )
        {
          _BitScanReverse(&v58, v34 - 1);
          v57 = 1 << (33 - (v58 ^ 0x1F));
          if ( v57 < 64 )
            v57 = 64;
        }
        v59 = *(_DWORD **)(a1 + 144);
        if ( v56 == v57 )
        {
          *(_QWORD *)(a1 + 152) = 0;
          v66 = &v59[8 * v56];
          do
          {
            if ( v59 )
              *v59 = -1;
            v59 += 8;
          }
          while ( v66 != v59 );
        }
        else
        {
          j___libc_free_0(v59);
          v60 = (4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1);
          v61 = (((v60 >> 2) | v60) >> 4) | (v60 >> 2) | v60;
          v62 = ((((v61 >> 8) | v61) >> 16) | (v61 >> 8) | v61) + 1;
          *(_DWORD *)(a1 + 160) = v62;
          v63 = (_DWORD *)sub_22077B0(32 * v62);
          v64 = *(unsigned int *)(a1 + 160);
          *(_QWORD *)(a1 + 152) = 0;
          *(_QWORD *)(a1 + 144) = v63;
          for ( j = &v63[8 * v64]; j != v63; v63 += 8 )
          {
            if ( v63 )
              *v63 = -1;
          }
        }
      }
      else
      {
        if ( !v56 )
          goto LABEL_55;
        j___libc_free_0(*(_QWORD *)(a1 + 144));
        *(_QWORD *)(a1 + 144) = 0;
        *(_QWORD *)(a1 + 152) = 0;
        *(_DWORD *)(a1 + 160) = 0;
      }
    }
    else
    {
      for ( ; v36 != v38; v36 += 32 )
      {
        if ( *(_DWORD *)v36 != -1 )
        {
          if ( *(_DWORD *)v36 != -2 )
          {
            v39 = *(_QWORD *)(v36 + 8);
            if ( v39 )
              j_j___libc_free_0(v39, *(_QWORD *)(v36 + 24) - v39);
          }
          *(_DWORD *)v36 = -1;
        }
      }
LABEL_55:
      *(_QWORD *)(a1 + 152) = 0;
    }
  }
LABEL_56:
  v40 = 0;
  v41 = 8LL * (unsigned int)v83;
  if ( (_DWORD)v83 )
  {
    v42 = a1;
    do
    {
      while ( 1 )
      {
        v43 = *(_QWORD *)&v82[v40];
        if ( **(_WORD **)(v43 + 16) == 15 )
        {
          v44 = *(_QWORD *)(v43 + 32);
          v45 = *(_DWORD *)(v44 + 48);
          v75 = *(_DWORD *)(v44 + 8);
          if ( (unsigned __int8)sub_1E69E00(*(_QWORD *)(v42 + 32), v45) )
          {
            if ( sub_1E69410(
                   *(__int64 **)(v42 + 32),
                   v45,
                   *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v42 + 32) + 24LL) + 16LL * (v75 & 0x7FFFFFFF))
                 & 0xFFFFFFFFFFFFFFF8LL,
                   0) )
            {
              break;
            }
          }
        }
        v40 += 8;
        if ( v41 == v40 )
          goto LABEL_63;
      }
      v40 += 8;
      sub_1E69BA0(*(_QWORD **)(v42 + 32), v75, v45);
      sub_1E16240(v43);
    }
    while ( v41 != v40 );
  }
LABEL_63:
  if ( a5 )
  {
    v46 = (unsigned __int64)v77;
    if ( v77 == v79 )
    {
      v48 = (unsigned int)v78;
      v49 = *(unsigned int *)(a5 + 8);
      v50 = v78;
      if ( (unsigned int)v78 <= v49 )
      {
        if ( (_DWORD)v78 )
          memmove(*(void **)a5, v79, 8LL * (unsigned int)v78);
      }
      else
      {
        if ( (unsigned int)v78 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
        {
          *(_DWORD *)(a5 + 8) = 0;
          sub_16CD150(a5, (const void *)(a5 + 16), v48, 8, v13, v14);
          v51 = v77;
          v48 = (unsigned int)v78;
          v49 = 0;
          v53 = v77;
        }
        else
        {
          v51 = v79;
          v52 = 8 * v49;
          v53 = v79;
          if ( *(_DWORD *)(a5 + 8) )
          {
            memmove(*(void **)a5, v79, v52);
            v51 = v77;
            v48 = (unsigned int)v78;
            v49 = v52;
            v53 = &v77[v52];
          }
        }
        v54 = 8 * v48;
        if ( v53 != &v51[v54] )
          memcpy((void *)(v49 + *(_QWORD *)a5), v53, v54 - v49);
      }
      LODWORD(v78) = 0;
      *(_DWORD *)(a5 + 8) = v50;
    }
    else
    {
      if ( *(_QWORD *)a5 != a5 + 16 )
      {
        _libc_free(*(_QWORD *)a5);
        v46 = (unsigned __int64)v77;
      }
      *(_QWORD *)a5 = v46;
      v47 = v78;
      v78 = 0;
      *(_QWORD *)(a5 + 8) = v47;
      v77 = v79;
    }
  }
  sub_2107430(v76);
  if ( (_BYTE *)v80[0] != v81 )
    _libc_free(v80[0]);
LABEL_4:
  if ( v82 != v84 )
    _libc_free((unsigned __int64)v82);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
  if ( (_BYTE *)v88[0] != v89 )
    _libc_free(v88[0]);
  if ( (v86 & 1) == 0 )
    j___libc_free_0(v87);
  return v67;
}
