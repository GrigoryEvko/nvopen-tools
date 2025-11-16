// Function: sub_2F5FFA0
// Address: 0x2f5ffa0
//
void __fastcall sub_2F5FFA0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r15
  __int64 v4; // r14
  __int64 v5; // rax
  unsigned int v6; // r12d
  __int64 v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // rcx
  _DWORD *v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 i; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned __int16 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 j; // rdx
  unsigned __int16 *v28; // rax
  unsigned __int64 v29; // rdi
  unsigned __int16 v30; // dx
  __int64 v31; // rsi
  unsigned __int64 v32; // rax
  int v33; // ebx
  unsigned int v34; // r12d
  __int64 v35; // rdx
  unsigned __int16 v36; // ax
  __int64 v37; // r15
  char *v38; // rax
  char *v39; // r14
  char *k; // rbx
  char *v41; // rax
  __int64 (*v42)(); // rax
  unsigned int v43; // eax
  __int64 v44; // rbx
  __int64 v45; // rsi
  __int64 (*v46)(); // rdx
  int v47; // eax
  __int64 v48; // rdx
  _QWORD *v49; // rcx
  __int64 v50; // rbx
  unsigned __int64 v51; // r12
  __int64 v52; // rax
  unsigned __int64 v53; // r8
  void *v54; // rdi
  size_t v55; // rdx
  size_t v56; // rdx
  __int64 v57; // rdx
  char v58; // [rsp+17h] [rbp-99h]
  __int16 s; // [rsp+18h] [rbp-98h]
  _QWORD *v60; // [rsp+20h] [rbp-90h]
  unsigned __int16 *v61; // [rsp+28h] [rbp-88h]
  void *s2; // [rsp+30h] [rbp-80h] BYREF
  __int64 v63; // [rsp+38h] [rbp-78h]
  _BYTE v64[48]; // [rsp+40h] [rbp-70h] BYREF
  int v65; // [rsp+70h] [rbp-40h]

  v2 = a1;
  a1[2] = a2;
  v4 = *(_QWORD *)(a2 + 16);
  if ( a1[3] == (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 200LL))(v4) )
  {
    v28 = sub_2EBFBC0(*(_QWORD **)(a1[2] + 32LL));
    v29 = a1[5];
    v30 = *v28;
    v61 = v28;
    if ( *v28 )
    {
      if ( v29 )
      {
        v16 = v2[4];
        v17 = (__int64)v28;
        v15 = 0;
        v31 = 0;
        while ( *(_WORD *)(v16 + v31) == v30 )
        {
          v32 = (unsigned int)(v15 + 1);
          v30 = v61[v32];
          v15 = v32;
          v31 = 2 * v32;
          if ( !v30 )
            goto LABEL_35;
          if ( v32 >= v29 )
            break;
        }
      }
    }
    else
    {
      v32 = 0;
LABEL_35:
      if ( v29 == v32 )
      {
        v58 = 0;
        goto LABEL_38;
      }
    }
  }
  else
  {
    v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 200LL))(v4);
    a1[3] = v5;
    v6 = (__int64)(*(_QWORD *)(v5 + 288) - *(_QWORD *)(v5 + 280)) >> 3;
    v7 = 3LL * v6;
    v8 = (_QWORD *)sub_2207820(v7 * 8 + 8);
    if ( v8 )
    {
      *v8 = v6;
      v9 = v8 + 1;
      if ( v6 )
      {
        v10 = v8 + 1;
        do
        {
          *v10 = 0;
          v10 += 6;
          *(v10 - 5) = 0;
          *((_BYTE *)v10 - 16) = 0;
          *((_BYTE *)v10 - 15) = 0;
          *((_WORD *)v10 - 7) = 0;
          *((_QWORD *)v10 - 1) = 0;
        }
        while ( v10 != (_DWORD *)&v9[v7] );
      }
    }
    else
    {
      v9 = 0;
    }
    v11 = *a1;
    *a1 = v9;
    if ( v11 )
    {
      v12 = 24LL * *(_QWORD *)(v11 - 8);
      for ( i = v11 + v12; v11 != i; i -= 24 )
      {
        v14 = *(_QWORD *)(i - 8);
        if ( v14 )
          j_j___libc_free_0_0(v14);
      }
      j_j_j___libc_free_0_0(v11 - 8);
    }
    v61 = sub_2EBFBC0(*(_QWORD **)(v2[2] + 32LL));
  }
  v18 = v2[3];
  v2[5] = 0;
  v19 = *(unsigned int *)(v18 + 44);
  if ( v19 > v2[13] )
  {
    v57 = *(unsigned int *)(v18 + 44);
    v2[12] = 0;
    sub_C8D290((__int64)(v2 + 11), v2 + 14, v57, 2u, v16, v17);
    if ( v19 )
      memset((void *)v2[11], 0, 2 * v19);
  }
  else
  {
    v20 = v2[12];
    v21 = v20;
    if ( v19 <= v20 )
      v21 = v19;
    if ( v21 )
    {
      memset((void *)v2[11], 0, 2 * v21);
      v20 = v2[12];
    }
    if ( v19 > v20 && 2 * (v19 - v20) )
      memset((void *)(v2[11] + 2 * v20), 0, 2 * (v19 - v20));
  }
  v22 = v61;
  v2[12] = v19;
  if ( *v61 )
  {
    v16 = *v61;
    do
    {
      v24 = v2[3];
      v25 = *(_QWORD *)(v24 + 8);
      v26 = *(_DWORD *)(v25 + 24LL * (unsigned __int16)v16 + 16) >> 12;
      v15 = *(_DWORD *)(v25 + 24LL * (unsigned __int16)v16 + 16) & 0xFFF;
      for ( j = *(_QWORD *)(v24 + 56) + 2 * v26; j; v15 = (unsigned int)(*(__int16 *)(j - 2) + (_DWORD)v15) )
      {
        j += 2;
        *(_WORD *)(v2[11] + 2LL * (unsigned int)v15) = v16;
        if ( !*(_WORD *)(j - 2) )
        {
          v16 = *v22;
          break;
        }
        v16 = *v22;
      }
      v23 = v2[5];
      if ( (unsigned __int64)(v23 + 1) > v2[6] )
      {
        s = v16;
        sub_C8D290((__int64)(v2 + 4), v2 + 7, v23 + 1, 2u, v16, v17);
        v23 = v2[5];
        LOWORD(v16) = s;
      }
      ++v22;
      *(_WORD *)(v2[4] + 2 * v23) = v16;
      ++v2[5];
      v16 = *v22;
    }
    while ( (_WORD)v16 );
  }
  v58 = 1;
LABEL_38:
  v33 = *(_DWORD *)(v2[3] + 16LL);
  s2 = v64;
  v34 = (unsigned int)(v33 + 63) >> 6;
  v63 = 0x600000000LL;
  v35 = v34;
  if ( v34 > 6 )
  {
    sub_C8D5F0((__int64)&s2, v64, v34, 8u, v16, v17);
    memset(s2, 0, 8LL * v34);
    LODWORD(v63) = (unsigned int)(v33 + 63) >> 6;
  }
  else
  {
    if ( v34 )
    {
      v35 = 8LL * v34;
      if ( v35 )
        memset(v64, 0, v35);
    }
    LODWORD(v63) = (unsigned int)(v33 + 63) >> 6;
  }
  v65 = v33;
  v36 = *v61;
  if ( *v61 )
  {
    v60 = v2;
    v37 = v4;
    do
    {
      v38 = sub_E922F0((_QWORD *)v60[3], v36);
      v39 = &v38[2 * v35];
      for ( k = v38; v39 != k; *(_QWORD *)v41 |= 1LL << v15 )
      {
        while ( 1 )
        {
          v35 = *(unsigned __int16 *)k;
          v42 = *(__int64 (**)())(*(_QWORD *)v37 + 456LL);
          if ( v42 != sub_2F5FDF0 )
            break;
          v41 = (char *)s2 + 8 * ((unsigned int)v35 >> 6);
          LODWORD(v35) = v35 & 0x3F;
LABEL_46:
          v15 = (unsigned int)v35;
          k += 2;
          v35 = ~(1LL << v35);
          *(_QWORD *)v41 &= v35;
          if ( v39 == k )
            goto LABEL_50;
        }
        v43 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v42)(v37, a2, v35, v35);
        v15 = *(unsigned __int16 *)k;
        v16 = v43;
        v35 = *(_WORD *)k & 0x3F;
        v41 = (char *)s2 + ((v15 >> 3) & 0x1FF8);
        if ( !(_BYTE)v16 )
          goto LABEL_46;
        k += 2;
      }
LABEL_50:
      v36 = *++v61;
    }
    while ( *v61 );
    v2 = v60;
    v33 = v65;
  }
  if ( v33 != *((_DWORD *)v2 + 54)
    || (v55 = 8LL * *((unsigned int *)v2 + 40)) != 0 && memcmp((const void *)v2[19], s2, v55) )
  {
    sub_2F5FE00((__int64)(v2 + 19), (__int64)&s2, v35, v15, v16, v17);
    v58 = 1;
    *((_DWORD *)v2 + 54) = v65;
  }
  v44 = v2[3];
  v45 = v2[2];
  v46 = *(__int64 (**)())(*(_QWORD *)v44 + 328LL);
  v47 = 0;
  if ( v46 != sub_2F3F790 )
  {
    v47 = ((__int64 (__fastcall *)(_QWORD, __int64))v46)(v2[3], v45);
    v45 = v2[2];
  }
  v48 = *(unsigned int *)(v44 + 16);
  v49 = *(_QWORD **)(v44 + 248);
  v2[38] = *v49 + (unsigned int)(v48 * v47);
  v2[39] = (unsigned int)v48;
  v50 = *(_QWORD *)(v45 + 32);
  if ( *(_DWORD *)(v50 + 448) == *((_DWORD *)v2 + 72)
    && ((v56 = 8LL * *(unsigned int *)(v50 + 392)) == 0
     || !memcmp(*(const void **)(v50 + 384), (const void *)v2[28], v56)) )
  {
    if ( !v58 )
      goto LABEL_63;
  }
  else
  {
    sub_2F5FE00((__int64)(v2 + 28), v50 + 384, v48, (__int64)v49, v16, v17);
    *((_DWORD *)v2 + 72) = *(_DWORD *)(v50 + 448);
  }
  v51 = 4LL * (*(unsigned int (__fastcall **)(_QWORD))(*(_QWORD *)v2[3] + 392LL))(v2[3]);
  v52 = sub_2207820(v51);
  v53 = v2[37];
  v54 = (void *)v52;
  v2[37] = v52;
  if ( v53 )
  {
    j_j___libc_free_0_0(v53);
    v54 = (void *)v2[37];
  }
  if ( v51 )
    memset(v54, 0, v51);
  ++*((_DWORD *)v2 + 2);
LABEL_63:
  if ( s2 != v64 )
    _libc_free((unsigned __int64)s2);
}
