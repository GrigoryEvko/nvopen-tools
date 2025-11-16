// Function: sub_28FB6F0
// Address: 0x28fb6f0
//
void __fastcall sub_28FB6F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // r13
  __int64 v8; // rdx
  __int64 *v9; // r12
  __int64 *v10; // r15
  __int64 *v11; // rax
  __int64 v12; // r14
  int v13; // ecx
  unsigned __int8 *v14; // rax
  bool v15; // r12
  int v16; // ecx
  int v17; // ecx
  unsigned int v18; // edx
  __int64 *v19; // r15
  unsigned __int8 *v20; // rdi
  __int64 v21; // rax
  __int64 *v22; // rax
  int v23; // edx
  int v24; // ecx
  __int64 v25; // rdi
  unsigned int v26; // edx
  _QWORD *v27; // r12
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int8 **v34; // rdi
  unsigned __int8 **v35; // r14
  unsigned __int8 **v36; // r13
  unsigned __int8 *v37; // rax
  unsigned __int8 *v38; // rdi
  unsigned __int8 *v39; // r12
  unsigned __int8 v40; // r15
  __int64 v41; // rax
  int v42; // edx
  unsigned int v43; // edx
  bool v44; // zf
  char *v45; // rax
  int v46; // r10d
  int v47; // r8d
  int v48; // r8d
  bool v49; // [rsp+1Fh] [rbp-141h]
  _QWORD v50[2]; // [rsp+20h] [rbp-140h] BYREF
  unsigned __int8 *v51; // [rsp+30h] [rbp-130h]
  __int64 v52[3]; // [rsp+40h] [rbp-120h] BYREF
  unsigned __int64 *v53; // [rsp+58h] [rbp-108h]
  __int64 v54; // [rsp+60h] [rbp-100h] BYREF
  __int64 v55; // [rsp+68h] [rbp-F8h]
  unsigned __int8 *v56; // [rsp+70h] [rbp-F0h]
  unsigned __int8 **v57; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v58; // [rsp+88h] [rbp-D8h]
  _BYTE v59[64]; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v60; // [rsp+D0h] [rbp-90h] BYREF
  char *v61; // [rsp+D8h] [rbp-88h]
  __int64 v62; // [rsp+E0h] [rbp-80h]
  unsigned __int64 *v63; // [rsp+E8h] [rbp-78h]
  char v64; // [rsp+F0h] [rbp-70h] BYREF

  v6 = (unsigned __int8 *)a2;
  v8 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( (*((_BYTE *)a2 + 7) & 0x40) != 0 )
  {
    v9 = (__int64 *)*(a2 - 1);
    v10 = &v9[(unsigned __int64)v8 / 8];
  }
  else
  {
    v10 = a2;
    v9 = &a2[v8 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v11 = (__int64 *)v59;
  v58 = 0x800000000LL;
  v12 = v8 >> 5;
  v13 = 0;
  v57 = (unsigned __int8 **)v59;
  if ( (unsigned __int64)v8 > 0x100 )
  {
    a2 = (__int64 *)v59;
    sub_C8D5F0((__int64)&v57, v59, v8 >> 5, 8u, a5, a6);
    v13 = v58;
    v11 = (__int64 *)&v57[(unsigned int)v58];
  }
  if ( v9 != v10 )
  {
    do
    {
      if ( v11 )
        *v11 = *v9;
      v9 += 4;
      ++v11;
    }
    while ( v9 != v10 );
    v13 = v58;
  }
  v54 = 0;
  v14 = v6;
  LODWORD(v58) = v12 + v13;
  v56 = v6;
  v55 = 0;
  v15 = v6 + 0x2000 != 0 && v6 + 4096 != 0;
  if ( v15 )
  {
    sub_BD73F0((__int64)&v54);
    v14 = v56;
  }
  v16 = *(_DWORD *)(a1 + 56);
  if ( v16 )
  {
    v17 = v16 - 1;
    a2 = *(__int64 **)(a1 + 40);
    v60 = 0;
    v62 = -4096;
    v61 = 0;
    v18 = v17 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v19 = &a2[4 * v18];
    v20 = (unsigned __int8 *)v19[2];
    if ( v14 == v20 )
    {
LABEL_14:
      sub_D68D70(&v60);
      v62 = -8192;
      v60 = 0;
      v61 = 0;
      v21 = v19[2];
      if ( v21 != -8192 )
      {
        if ( v21 != -4096 && v21 )
          sub_BD60C0(v19);
        v19[2] = -8192;
      }
      sub_D68D70(&v60);
      --*(_DWORD *)(a1 + 48);
      v14 = v56;
      ++*(_DWORD *)(a1 + 52);
    }
    else
    {
      v47 = 1;
      while ( v20 != (unsigned __int8 *)-4096LL )
      {
        v18 = v17 & (v47 + v18);
        v19 = &a2[4 * v18];
        v20 = (unsigned __int8 *)v19[2];
        if ( v14 == v20 )
          goto LABEL_14;
        ++v47;
      }
      sub_D68D70(&v60);
      v14 = v56;
    }
  }
  if ( v14 != 0 && v14 + 4096 != 0 && v14 != (unsigned __int8 *)-8192LL )
    sub_BD60C0(&v54);
  v51 = v6;
  v22 = (__int64 *)v6;
  v50[0] = 0;
  v50[1] = 0;
  if ( v15 )
  {
    sub_BD73F0((__int64)v50);
    v22 = (__int64 *)v51;
  }
  v23 = *(_DWORD *)(a1 + 88);
  if ( v23 )
  {
    v24 = v23 - 1;
    v25 = *(_QWORD *)(a1 + 72);
    v60 = 0;
    v62 = -4096;
    v61 = 0;
    v26 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v27 = (_QWORD *)(v25 + 24LL * v26);
    a2 = (__int64 *)v27[2];
    if ( v22 == a2 )
    {
LABEL_27:
      sub_D68D70(&v60);
      v62 = -8192;
      v60 = 0;
      v61 = 0;
      v28 = v27[2];
      if ( v28 != -8192 )
      {
        if ( v28 && v28 != -4096 )
          sub_BD60C0(v27);
        v27[2] = -8192;
      }
      sub_D68D70(&v60);
      --*(_DWORD *)(a1 + 80);
      ++*(_DWORD *)(a1 + 84);
      sub_28ED760(v52, (_QWORD *)(a1 + 96), (__int64)v50);
      LODWORD(a2) = a1 + 96;
      v60 = v52[0];
      v29 = *v53;
      v63 = v53;
      v61 = (char *)v29;
      v62 = v29 + 504;
      sub_28FB170(&v54, (_QWORD *)(a1 + 96), (__int64)&v60);
      v22 = (__int64 *)v51;
    }
    else
    {
      v48 = 1;
      while ( a2 != (__int64 *)-4096LL )
      {
        v26 = v24 & (v48 + v26);
        v27 = (_QWORD *)(v25 + 24LL * v26);
        a2 = (__int64 *)v27[2];
        if ( v22 == a2 )
          goto LABEL_27;
        ++v48;
      }
      sub_D68D70(&v60);
      v22 = (__int64 *)v51;
    }
  }
  if ( v22 != 0 && v22 + 512 != 0 && v22 != (__int64 *)-8192LL )
    sub_BD60C0(v50);
  sub_F54ED0(v6);
  sub_B43D60(v6);
  BYTE4(v63) = 1;
  v34 = v57;
  v61 = &v64;
  v60 = 0;
  v35 = v57;
  v36 = &v57[(unsigned int)v58];
  v62 = 8;
  LODWORD(v63) = 0;
  if ( v57 == v36 )
  {
    *(_BYTE *)(a1 + 752) = 1;
    goto LABEL_56;
  }
  do
  {
    while ( 1 )
    {
      v39 = *v35;
      v40 = **v35;
      if ( v40 <= 0x1Cu )
        goto LABEL_47;
      v41 = *((_QWORD *)v39 + 2);
      if ( v41 )
      {
        while ( !*(_QWORD *)(v41 + 8) && v40 == **(_BYTE **)(v41 + 24) )
        {
          if ( !BYTE4(v63) )
            goto LABEL_61;
          v45 = v61;
          v31 = HIDWORD(v62);
          v30 = (__int64)&v61[8 * HIDWORD(v62)];
          if ( v61 != (char *)v30 )
          {
            while ( v39 != *(unsigned __int8 **)v45 )
            {
              v45 += 8;
              if ( (char *)v30 == v45 )
                goto LABEL_69;
            }
            break;
          }
LABEL_69:
          if ( HIDWORD(v62) < (unsigned int)v62 )
          {
            v31 = (unsigned int)++HIDWORD(v62);
            *(_QWORD *)v30 = v39;
            ++v60;
          }
          else
          {
LABEL_61:
            LODWORD(a2) = (_DWORD)v39;
            sub_C8CC70((__int64)&v60, (__int64)v39, v30, v31, v32, v33);
            if ( !(_BYTE)v30 )
              break;
          }
          v39 = *(unsigned __int8 **)(*((_QWORD *)v39 + 2) + 24LL);
          v41 = *((_QWORD *)v39 + 2);
          if ( !v41 )
            break;
        }
      }
      v56 = v39;
      v54 = 0;
      v55 = 0;
      v49 = v39 + 4096 != 0 && v39 + 0x2000 != 0;
      if ( v49 )
        break;
      v32 = *(unsigned int *)(a1 + 56);
      v30 = 0;
      v37 = v39;
      if ( (_DWORD)v32 )
        goto LABEL_39;
LABEL_47:
      if ( v36 == ++v35 )
        goto LABEL_54;
    }
    sub_BD73F0((__int64)&v54);
    v37 = v56;
    v32 = *(unsigned int *)(a1 + 56);
    LOBYTE(v42) = v56 + 0x2000 != 0;
    LOBYTE(a2) = v56 + 4096 != 0;
    v43 = (unsigned int)a2 & v42;
    LOBYTE(a2) = v56 != 0;
    v30 = (unsigned int)a2 & v43;
    if ( (_DWORD)v32 )
    {
LABEL_39:
      v32 = (unsigned int)(v32 - 1);
      v33 = *(_QWORD *)(a1 + 40);
      LODWORD(a2) = v32 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v38 = *(unsigned __int8 **)(v33 + 32LL * (unsigned int)a2 + 16);
      if ( v37 == v38 )
      {
LABEL_40:
        if ( (_BYTE)v30 )
          sub_BD60C0(&v54);
        v56 = v39;
        v54 = 0;
        v55 = 0;
        if ( v49 )
          sub_BD73F0((__int64)&v54);
        a2 = &v54;
        sub_28F19A0(a1 + 64, &v54);
        LOBYTE(v30) = v56 != 0;
        if ( v56 == 0 || v56 + 4096 == 0 || v56 == (unsigned __int8 *)-8192LL )
          goto LABEL_47;
      }
      else
      {
        v46 = 1;
        while ( v38 != (unsigned __int8 *)-4096LL )
        {
          LODWORD(a2) = v32 & (v46 + (_DWORD)a2);
          v31 = 32LL * (unsigned int)a2;
          v38 = *(unsigned __int8 **)(v33 + v31 + 16);
          if ( v38 == v37 )
            goto LABEL_40;
          ++v46;
        }
        if ( !(_BYTE)v30 )
          goto LABEL_47;
      }
LABEL_46:
      sub_BD60C0(&v54);
      goto LABEL_47;
    }
    if ( (_BYTE)v30 )
      goto LABEL_46;
    ++v35;
  }
  while ( v36 != v35 );
LABEL_54:
  v44 = BYTE4(v63) == 0;
  *(_BYTE *)(a1 + 752) = 1;
  v34 = v57;
  if ( v44 )
  {
    _libc_free((unsigned __int64)v61);
    v34 = v57;
  }
LABEL_56:
  if ( v34 != (unsigned __int8 **)v59 )
    _libc_free((unsigned __int64)v34);
}
