// Function: sub_1A073B0
// Address: 0x1a073b0
//
__int64 __fastcall sub_1A073B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        double a4,
        double a5,
        double a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  __int64 v9; // r15
  unsigned int v10; // r14d
  char *v11; // rdx
  __int64 v12; // r12
  unsigned int i; // r15d
  __int64 v14; // rbx
  int v15; // eax
  __int64 v16; // r11
  __int64 v17; // r11
  __int64 *v18; // rcx
  _QWORD *v19; // r10
  __int64 v20; // rdx
  __int64 *v21; // r8
  __int64 v22; // rax
  _QWORD *v23; // rcx
  __int64 j; // r15
  __int64 v25; // r13
  char *v26; // rsi
  char *v27; // rbx
  int v28; // ecx
  char *v29; // rdi
  __int64 v30; // rbx
  __int64 v31; // r14
  unsigned __int64 v32; // rbx
  char *v33; // r12
  __int64 v34; // r14
  char *v35; // r12
  unsigned int v36; // eax
  __int64 v37; // rax
  char *v38; // r12
  int v39; // eax
  _BYTE *v40; // rdi
  __int64 v41; // r12
  char *v43; // rax
  char *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rbx
  int v48; // r8d
  int v49; // r9d
  __int64 v50; // rax
  __int64 v51; // rax
  _QWORD *v53; // [rsp+10h] [rbp-D0h]
  __int64 v54; // [rsp+18h] [rbp-C8h]
  __int64 *v55; // [rsp+20h] [rbp-C0h]
  __int64 v57; // [rsp+38h] [rbp-A8h]
  __int64 v58; // [rsp+48h] [rbp-98h] BYREF
  _QWORD *v59; // [rsp+50h] [rbp-90h] BYREF
  __int64 v60; // [rsp+58h] [rbp-88h]
  _BYTE v61[32]; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v62; // [rsp+80h] [rbp-60h] BYREF
  __int64 v63; // [rsp+88h] [rbp-58h]
  _QWORD v64[10]; // [rsp+90h] [rbp-50h] BYREF

  v9 = a3;
  v10 = *(_DWORD *)(a3 + 8);
  v59 = v61;
  v11 = *(char **)a3;
  v60 = 0x400000000LL;
  if ( v10 > 1 )
  {
    LODWORD(a8) = 0;
    v57 = v9;
    LODWORD(v12) = 0;
    for ( i = 1; i < v10; ++i )
    {
      v14 = (__int64)&v11[16 * i];
      v15 = *(_DWORD *)(v14 + 8);
      if ( !v15 )
        break;
      v16 = (unsigned int)v12;
      LODWORD(v12) = i;
      v17 = 16 * v16;
      v18 = (__int64 *)&v11[v17];
      if ( v15 == *(_DWORD *)&v11[v17 + 8] )
      {
        v19 = (_QWORD *)v57;
        v20 = 1;
        v63 = 0x400000000LL;
        v21 = (__int64 *)&v62;
        v62 = v64;
        v22 = *v18;
        v23 = v64;
        v64[0] = v22;
        v12 = i + 1;
        LODWORD(v63) = 1;
        for ( j = 16 * v12; ; j += 16 )
        {
          v23[v20] = *(_QWORD *)v14;
          v20 = (unsigned int)(v63 + 1);
          LODWORD(v63) = v63 + 1;
          if ( (unsigned int)v12 >= v10 )
          {
            i = v12;
            v25 = *(_QWORD *)v57 + v17;
            goto LABEL_43;
          }
          v14 = *v19 + j;
          v25 = v17 + *v19;
          if ( *(_DWORD *)(v14 + 8) != *(_DWORD *)(v25 + 8) )
            break;
          if ( HIDWORD(v63) <= (unsigned int)v20 )
          {
            v53 = v19;
            v54 = v17;
            v55 = v21;
            sub_16CD150((__int64)v21, v64, 0, 8, (int)v21, a9);
            v20 = (unsigned int)v63;
            v19 = v53;
            v17 = v54;
            v21 = v55;
          }
          v23 = v62;
          LODWORD(v12) = v12 + 1;
        }
        i = v12;
LABEL_43:
        v45 = sub_19FF410(a2, v21, a4, a5, a6);
        *(_QWORD *)v25 = v45;
        if ( *(_BYTE *)(v45 + 16) > 0x17u )
        {
          v58 = v45;
          sub_1A062A0(a1 + 64, &v58);
        }
        if ( v62 != v64 )
          _libc_free((unsigned __int64)v62);
        v11 = *(char **)v57;
      }
    }
    v9 = v57;
    v10 = *(_DWORD *)(v57 + 8);
  }
  v26 = &v11[16 * v10];
  if ( v26 != v11 )
  {
    v27 = v11;
    do
    {
      v29 = v27;
      v27 += 16;
      if ( v26 == v27 )
        goto LABEL_17;
      v28 = *((_DWORD *)v27 - 2);
    }
    while ( v28 != *((_DWORD *)v27 + 2) );
    if ( v26 == v29 )
    {
      v27 = &v11[16 * v10];
    }
    else
    {
      v43 = v29 + 32;
      if ( v26 != v29 + 32 )
      {
        while ( 1 )
        {
          if ( *((_DWORD *)v43 + 2) != v28 )
          {
            v29 += 16;
            *(_QWORD *)v29 = *(_QWORD *)v43;
            *((_DWORD *)v29 + 2) = *((_DWORD *)v43 + 2);
          }
          v43 += 16;
          if ( v26 == v43 )
            break;
          v28 = *((_DWORD *)v29 + 2);
        }
        v11 = *(char **)v9;
        v44 = v29 + 16;
        a8 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8) - (_QWORD)v26;
        v27 = &v44[a8];
        if ( v26 != (char *)(*(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8)) )
        {
          memmove(v44, v26, *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8) - (_QWORD)v26);
          v11 = *(char **)v9;
        }
      }
    }
LABEL_17:
    v30 = (v27 - v11) >> 4;
    *(_DWORD *)(v9 + 8) = v30;
    if ( !(_DWORD)v30 )
      goto LABEL_26;
    v31 = (unsigned int)(v30 - 1);
    v32 = 0;
    v33 = v11;
    v34 = 16 * v31;
    while ( 1 )
    {
      v35 = &v33[v32];
      v36 = *((_DWORD *)v35 + 2);
      if ( (v36 & 1) != 0 )
      {
        v37 = (unsigned int)v60;
        if ( (unsigned int)v60 >= HIDWORD(v60) )
        {
          sub_16CD150((__int64)&v59, v61, 0, 8, a8, a9);
          v37 = (unsigned int)v60;
        }
        v59[v37] = *(_QWORD *)v35;
        v38 = *(char **)v9;
        LODWORD(v60) = v60 + 1;
        *(_DWORD *)&v38[v32 + 8] >>= 1;
        if ( v34 == v32 )
        {
LABEL_25:
          v11 = *(char **)v9;
          goto LABEL_26;
        }
      }
      else
      {
        *((_DWORD *)v35 + 2) = v36 >> 1;
        if ( v34 == v32 )
          goto LABEL_25;
      }
      v33 = *(char **)v9;
      v32 += 16LL;
    }
  }
  *(_DWORD *)(v9 + 8) = 0;
LABEL_26:
  if ( *((_DWORD *)v11 + 2) )
  {
    v47 = sub_1A073B0(a1, a2, v9);
    v50 = (unsigned int)v60;
    if ( (unsigned int)v60 >= HIDWORD(v60) )
    {
      sub_16CD150((__int64)&v59, v61, 0, 8, v48, v49);
      v50 = (unsigned int)v60;
    }
    v59[v50] = v47;
    v51 = (unsigned int)(v60 + 1);
    LODWORD(v60) = v51;
    if ( HIDWORD(v60) <= (unsigned int)v51 )
    {
      sub_16CD150((__int64)&v59, v61, 0, 8, v48, v49);
      v51 = (unsigned int)v60;
    }
    v59[v51] = v47;
    v39 = v60 + 1;
    LODWORD(v60) = v60 + 1;
  }
  else
  {
    v39 = v60;
  }
  if ( v39 == 1 )
  {
    v40 = v59;
    v41 = *v59;
  }
  else
  {
    v46 = sub_19FF410(a2, (__int64 *)&v59, a4, a5, a6);
    v40 = v59;
    v41 = v46;
  }
  if ( v40 != v61 )
    _libc_free((unsigned __int64)v40);
  return v41;
}
