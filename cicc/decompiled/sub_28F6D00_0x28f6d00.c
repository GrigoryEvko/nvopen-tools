// Function: sub_28F6D00
// Address: 0x28f6d00
//
__int64 __fastcall sub_28F6D00(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v4; // rax
  char v5; // al
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rdx
  int v9; // ebx
  int v10; // r12d
  _BYTE *v11; // rax
  __int64 v12; // r15
  __int64 v13; // r14
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned int v16; // r9d
  _BYTE *v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // rbx
  __int64 v21; // r13
  unsigned int v22; // eax
  unsigned __int64 v23; // rdx
  char *v24; // rdx
  unsigned int v25; // edx
  unsigned __int64 v26; // r15
  bool v27; // cc
  bool v28; // al
  bool v29; // r14
  _BYTE *v30; // rdi
  __int64 v31; // r15
  char *v32; // r14
  __int64 *v33; // rsi
  int v34; // eax
  __int64 v35; // r13
  _QWORD *i; // r15
  int v38; // edx
  _BYTE *v39; // rax
  int v40; // eax
  __int64 v41; // r12
  int v42; // eax
  _BYTE *v43; // rdx
  _BYTE *v44; // rsi
  __int64 v45; // [rsp+0h] [rbp-1E0h]
  __int64 v47; // [rsp+28h] [rbp-1B8h]
  __int64 v48; // [rsp+30h] [rbp-1B0h]
  unsigned int v49; // [rsp+30h] [rbp-1B0h]
  __int64 v50; // [rsp+38h] [rbp-1A8h]
  unsigned int v51; // [rsp+38h] [rbp-1A8h]
  unsigned int v52; // [rsp+4Ch] [rbp-194h] BYREF
  char *v53; // [rsp+50h] [rbp-190h] BYREF
  unsigned int v54; // [rsp+58h] [rbp-188h]
  char *v55; // [rsp+60h] [rbp-180h] BYREF
  _QWORD *v56; // [rsp+68h] [rbp-178h]
  char v57; // [rsp+80h] [rbp-160h]
  char v58; // [rsp+81h] [rbp-15Fh]
  _BYTE *v59; // [rsp+90h] [rbp-150h] BYREF
  __int64 v60; // [rsp+98h] [rbp-148h]
  _BYTE v61[128]; // [rsp+A0h] [rbp-140h] BYREF
  _BYTE *v62; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+128h] [rbp-B8h]
  _BYTE v64[176]; // [rsp+130h] [rbp-B0h] BYREF

  v4 = sub_28ED370(a2, 17, 18);
  v47 = (__int64)v4;
  if ( !v4 )
    return 0;
  v59 = v61;
  v45 = a1 + 64;
  v60 = 0x800000000LL;
  v52 = (unsigned int)&loc_1010101;
  v5 = sub_28F1DF0(v4, (__int64)&v59, a1 + 64, &v52);
  v8 = (unsigned int)v60;
  *(_BYTE *)(a1 + 752) |= v5;
  v63 = 0x800000000LL;
  v9 = v8;
  v62 = v64;
  if ( (unsigned int)v8 > 8 )
  {
    sub_C8D5F0((__int64)&v62, v64, v8, 0x10u, v6, v7);
    v9 = v60;
    if ( !(_DWORD)v60 )
    {
      v19 = v63;
      goto LABEL_15;
    }
  }
  else if ( !(_DWORD)v8 )
  {
LABEL_45:
    v35 = 0;
    sub_28F62D0(a1, (unsigned __int8 *)v47, (__int64)&v62, v52, v6);
    goto LABEL_46;
  }
  v50 = a3;
  v10 = 0;
  do
  {
    v11 = &v59[16 * v10];
    v12 = *(_QWORD *)v11;
    v13 = *((_QWORD *)v11 + 1);
    v14 = sub_28EF780(a1, *(_BYTE **)v11);
    v15 = (unsigned int)v63;
    v16 = v14;
    if ( v13 + (unsigned __int64)(unsigned int)v63 > HIDWORD(v63) )
    {
      v49 = v14;
      sub_C8D5F0((__int64)&v62, v64, v13 + (unsigned int)v63, 0x10u, v6, v14);
      v15 = (unsigned int)v63;
      v16 = v49;
    }
    v17 = &v62[16 * v15];
    if ( v13 )
    {
      v18 = v13;
      do
      {
        if ( v17 )
        {
          *(_DWORD *)v17 = v16;
          *((_QWORD *)v17 + 1) = v12;
        }
        v17 += 16;
        --v18;
      }
      while ( v18 );
      LODWORD(v15) = v63;
    }
    ++v10;
    LODWORD(v63) = v15 + v13;
    v19 = v15 + v13;
  }
  while ( v10 != v9 );
  a3 = v50;
LABEL_15:
  if ( !v19 )
    goto LABEL_45;
  v20 = 0;
  v48 = a3 + 24;
  v21 = 16LL * v19;
  while ( 1 )
  {
    v30 = &v62[v20];
    v31 = *(_QWORD *)&v62[v20 + 8];
    if ( v31 == a3 )
      break;
    if ( *(_BYTE *)a3 == 17 )
    {
      if ( *(_BYTE *)v31 != 17 )
        goto LABEL_31;
      v22 = *(_DWORD *)(v31 + 32);
      v54 = v22;
      if ( v22 > 0x40 )
      {
        sub_C43780((__int64)&v53, (const void **)(v31 + 24));
        v22 = v54;
        if ( v54 > 0x40 )
        {
          sub_C43D10((__int64)&v53);
          goto LABEL_23;
        }
        v23 = (unsigned __int64)v53;
      }
      else
      {
        v23 = *(_QWORD *)(v31 + 24);
      }
      v24 = (char *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~v23);
      if ( !v22 )
        v24 = 0;
      v53 = v24;
LABEL_23:
      sub_C46250((__int64)&v53);
      v25 = v54;
      v26 = (unsigned __int64)v53;
      v54 = 0;
      v27 = *(_DWORD *)(a3 + 32) <= 0x40u;
      LODWORD(v56) = v25;
      v55 = v53;
      if ( v27 )
      {
        v29 = *(_QWORD *)(a3 + 24) == (_QWORD)v53;
      }
      else
      {
        v51 = v25;
        v28 = sub_C43C50(v48, (const void **)&v55);
        v25 = v51;
        v29 = v28;
      }
      if ( v25 > 0x40 )
      {
        if ( v26 )
        {
          j_j___libc_free_0_0(v26);
          if ( v54 > 0x40 )
          {
            if ( v53 )
              j_j___libc_free_0_0((unsigned __int64)v53);
          }
        }
      }
      if ( v29 )
      {
        v42 = v63;
        v43 = &v62[16 * (unsigned int)v63];
        v44 = &v62[v20 + 16];
        if ( v43 != v44 )
        {
          memmove(&v62[v20], v44, v43 - v44);
          v42 = v63;
        }
        v40 = v42 - 1;
        LODWORD(v63) = v40;
        goto LABEL_64;
      }
LABEL_31:
      v20 += 16;
      if ( v21 == v20 )
        goto LABEL_45;
    }
    else
    {
      if ( *(_BYTE *)a3 != 18 || *(_BYTE *)v31 != 18 )
        goto LABEL_31;
      v32 = (char *)sub_C33340();
      v33 = (__int64 *)(v31 + 24);
      if ( *(char **)(v31 + 24) == v32 )
        sub_C3C790(&v55, (_QWORD **)v33);
      else
        sub_C33EB0(&v55, v33);
      if ( v55 == v32 )
        sub_C3CCB0((__int64)&v55);
      else
        sub_C34440((unsigned __int8 *)&v55);
      if ( *(char **)(a3 + 24) == v32 )
        v34 = sub_C3E510(v48, (__int64)&v55);
      else
        v34 = sub_C37950(v48, (__int64)&v55);
      if ( v34 == 1 )
      {
        v29 = 1;
        sub_28E9AE0((__int64)&v62, &v62[v20]);
        sub_91D830(&v55);
        v40 = v63;
        goto LABEL_64;
      }
      if ( v32 == v55 )
      {
        if ( v56 )
        {
          for ( i = &v56[3 * *(v56 - 1)]; v56 != i; sub_91D830(i) )
            i -= 3;
          j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
        }
        goto LABEL_31;
      }
      v20 += 16;
      sub_C338F0((__int64)&v55);
      if ( v21 == v20 )
        goto LABEL_45;
    }
  }
  v38 = v63;
  v39 = &v62[16 * (unsigned int)v63];
  if ( v39 != v30 + 16 )
  {
    memmove(v30, v30 + 16, v39 - (v30 + 16));
    v38 = v63;
  }
  v40 = v38 - 1;
  v29 = 0;
  LODWORD(v63) = v38 - 1;
LABEL_64:
  v41 = *(_QWORD *)(v47 + 32);
  if ( v40 == 1 )
  {
    sub_D68D20((__int64)&v55, 0, v47);
    sub_28F19A0(v45, &v55);
    sub_D68D70(&v55);
    v35 = *((_QWORD *)v62 + 1);
  }
  else
  {
    v35 = v47;
    sub_28F62D0(a1, (unsigned __int8 *)v47, (__int64)&v62, v52, v6);
  }
  if ( v29 )
  {
    v58 = 1;
    v55 = "neg";
    v57 = 3;
    v35 = sub_28E9340(v35, (__int64)&v55, v41, 0, 0, (_BYTE *)v47);
  }
LABEL_46:
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
  return v35;
}
