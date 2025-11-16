// Function: sub_29A3E20
// Address: 0x29a3e20
//
__int64 __fastcall sub_29A3E20(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // r15
  int v9; // ebx
  __int64 v10; // r9
  char v11; // r14
  int v12; // r15d
  __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rbx
  __int64 v35; // rax
  signed __int64 v36; // r14
  __int64 *v37; // rax
  int v38; // r14d
  unsigned __int16 v39; // bx
  bool v40; // zf
  __int64 v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // rcx
  char v44; // al
  __int64 v45; // rbx
  __int64 *v46; // r14
  __int64 *v47; // r15
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // r15
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  _QWORD *v55; // r15
  unsigned __int64 v56; // rbx
  unsigned __int64 v57; // r14
  unsigned __int64 v58; // rax
  __int64 v60; // rax
  __int64 v61; // [rsp+0h] [rbp-200h]
  bool v62; // [rsp+Fh] [rbp-1F1h]
  __int64 v64; // [rsp+20h] [rbp-1E0h]
  __int64 v65; // [rsp+38h] [rbp-1C8h]
  __int64 v66; // [rsp+40h] [rbp-1C0h]
  __int64 v67; // [rsp+48h] [rbp-1B8h]
  __int64 v68; // [rsp+48h] [rbp-1B8h]
  __int64 *v69; // [rsp+60h] [rbp-1A0h]
  __int64 v70; // [rsp+68h] [rbp-198h]
  __int64 v71; // [rsp+78h] [rbp-188h] BYREF
  void *v72[4]; // [rsp+80h] [rbp-180h] BYREF
  __int16 v73; // [rsp+A0h] [rbp-160h]
  _BYTE *v74; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v75; // [rsp+B8h] [rbp-148h]
  _BYTE v76[32]; // [rsp+C0h] [rbp-140h] BYREF
  char v77[8]; // [rsp+E0h] [rbp-120h] BYREF
  char *v78; // [rsp+E8h] [rbp-118h]
  char v79; // [rsp+F8h] [rbp-108h] BYREF
  unsigned __int64 v80; // [rsp+100h] [rbp-100h]
  __int64 *v81; // [rsp+140h] [rbp-C0h] BYREF
  unsigned __int64 v82; // [rsp+148h] [rbp-B8h]
  char v83[8]; // [rsp+150h] [rbp-B0h] BYREF
  char v84; // [rsp+158h] [rbp-A8h] BYREF
  _QWORD *v85; // [rsp+160h] [rbp-A0h]

  v3 = (__int64)a1;
  if ( *(a1 - 4) )
  {
    v4 = *(a1 - 3);
    *(_QWORD *)*(a1 - 2) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = *(a1 - 2);
  }
  *(a1 - 4) = a2;
  if ( a2 )
  {
    v5 = a2[2];
    *(a1 - 3) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = a1 - 3;
    *(a1 - 2) = a2 + 2;
    a2[2] = a1 - 4;
  }
  sub_B99FD0((__int64)a1, 2u, 0);
  sub_B99FD0((__int64)a1, 0x17u, 0);
  v6 = a2[3];
  if ( a1[10] != v6 )
  {
    v7 = a1[1];
    v8 = **(_QWORD **)(v6 + 16);
    a1[10] = v6;
    v64 = v7;
    a1[1] = v8;
    v61 = v8;
    v66 = a2[3];
    v9 = *(_DWORD *)(v66 + 12);
    v69 = (__int64 *)sub_B2BE50((__int64)a2);
    v62 = v8 != v7;
    v71 = a1[9];
    v74 = v76;
    v75 = 0x400000000LL;
    if ( v9 == 1 )
    {
      v60 = sub_A74610(&v71);
      sub_A74940((__int64)v77, (__int64)v69, v60);
      if ( *(_BYTE *)(v7 + 8) == 7 || v8 == v7 )
        goto LABEL_60;
    }
    else
    {
      v10 = 0;
      v11 = 0;
      v65 = (unsigned int)(v9 - 1);
      do
      {
        v12 = v10;
        v70 = v10 + 1;
        v13 = *(_QWORD *)(v3 + 32 * (v10 - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
        v14 = *(_QWORD *)(*(_QWORD *)(v66 + 16) + 8 * (v10 + 1));
        if ( *(_QWORD *)(v13 + 8) == v14 )
        {
          v52 = sub_A744E0(&v71, v10);
          v53 = (unsigned int)v75;
          v54 = (unsigned int)v75 + 1LL;
          if ( v54 > HIDWORD(v75) )
          {
            sub_C8D5F0((__int64)&v74, v76, v54, 8u, v50, v51);
            v53 = (unsigned int)v75;
          }
          *(_QWORD *)&v74[8 * v53] = v52;
          LODWORD(v75) = v75 + 1;
        }
        else
        {
          LOWORD(v85) = 257;
          v67 = v10;
          v15 = sub_B52260(v13, v14, (__int64)&v81, v3 + 24, 0);
          v16 = v3 + 32 * (v67 - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
          if ( *(_QWORD *)v16 )
          {
            v17 = *(_QWORD *)(v16 + 8);
            **(_QWORD **)(v16 + 16) = v17;
            if ( v17 )
              *(_QWORD *)(v17 + 16) = *(_QWORD *)(v16 + 16);
          }
          *(_QWORD *)v16 = v15;
          if ( v15 )
          {
            v18 = *(_QWORD *)(v15 + 16);
            *(_QWORD *)(v16 + 8) = v18;
            if ( v18 )
              *(_QWORD *)(v18 + 16) = v16 + 8;
            *(_QWORD *)(v16 + 16) = v15 + 16;
            *(_QWORD *)(v15 + 16) = v16;
          }
          v19 = sub_A744E0(&v71, v12);
          sub_A74940((__int64)&v81, (__int64)v69, v19);
          v20 = sub_A744E0(&v71, v12);
          sub_A751C0((__int64)v77, v14, v20, 3);
          sub_A74A10((__int64)&v81, (__int64)v77);
          v21 = v80;
          if ( v80 )
          {
            v68 = v3;
            do
            {
              v22 = v21;
              sub_29A3490(*(_QWORD **)(v21 + 24));
              v23 = *(_QWORD *)(v21 + 32);
              v21 = *(_QWORD *)(v21 + 16);
              if ( v23 != v22 + 56 )
                _libc_free(v23);
              j_j___libc_free_0(v22);
            }
            while ( v21 );
            v3 = v68;
          }
          if ( sub_A74E40((__int64)&v81, 81) )
          {
            v24 = sub_A748A0(a2 + 15, v12);
            sub_A77E90(&v81, v24);
          }
          if ( sub_A74E40((__int64)&v81, 83) )
          {
            v25 = sub_A74900(a2 + 15, v12);
            sub_A77EB0(&v81, v25);
          }
          v28 = sub_A7A280(v69, (__int64)&v81);
          v29 = (unsigned int)v75;
          v30 = (unsigned int)v75 + 1LL;
          if ( v30 > HIDWORD(v75) )
          {
            sub_C8D5F0((__int64)&v74, v76, v30, 8u, v26, v27);
            v29 = (unsigned int)v75;
          }
          *(_QWORD *)&v74[8 * v29] = v28;
          LODWORD(v75) = v75 + 1;
          if ( (char *)v82 != &v84 )
            _libc_free(v82);
          v11 = 1;
        }
        v10 = v70;
      }
      while ( v65 != v70 );
      v31 = sub_A74610(&v71);
      sub_A74940((__int64)v77, (__int64)v69, v31);
      if ( *(_BYTE *)(v64 + 8) == 7 || !v62 )
      {
        if ( !v11 )
        {
LABEL_60:
          if ( v78 != &v79 )
            _libc_free((unsigned __int64)v78);
          if ( v74 != v76 )
            _libc_free((unsigned __int64)v74);
          return v3;
        }
LABEL_59:
        v55 = v74;
        v56 = (unsigned int)v75;
        v57 = sub_A7A280(v69, (__int64)v77);
        v58 = sub_A74680(&v71);
        *(_QWORD *)(v3 + 72) = sub_A78180(v69, v58, v57, v55, v56);
        goto LABEL_60;
      }
    }
    v34 = *(_QWORD *)(v3 + 16);
    v81 = (__int64 *)v83;
    v82 = 0x1000000000LL;
    if ( v34 )
    {
      v35 = v34;
      v36 = 0;
      do
      {
        v35 = *(_QWORD *)(v35 + 8);
        ++v36;
      }
      while ( v35 );
      v37 = (__int64 *)v83;
      if ( v36 > 16 )
      {
        sub_C8D5F0((__int64)&v81, v83, v36, 8u, v32, v33);
        v37 = &v81[(unsigned int)v82];
      }
      do
      {
        *v37++ = *(_QWORD *)(v34 + 24);
        v34 = *(_QWORD *)(v34 + 8);
      }
      while ( v34 );
      v38 = v82 + v36;
    }
    else
    {
      v38 = 0;
    }
    HIBYTE(v39) = 0;
    v40 = *(_BYTE *)v3 == 34;
    LODWORD(v82) = v38;
    if ( v40 )
    {
      v41 = *(_QWORD *)(v3 - 96);
      v42 = *(_QWORD *)(v3 + 40);
      v73 = 257;
      v43 = *(_QWORD *)(sub_F41C30(v42, v41, 0, 0, 0, v72) + 56);
      v44 = 1;
    }
    else
    {
      v43 = *(_QWORD *)(v3 + 32);
      v44 = 0;
    }
    LOBYTE(v39) = v44;
    v73 = 257;
    v45 = sub_B52260(v3, v64, (__int64)v72, v43, v39);
    if ( a3 )
      *a3 = v45;
    v46 = &v81[(unsigned int)v82];
    if ( v81 != v46 )
    {
      v47 = v81;
      do
      {
        v48 = *v47++;
        sub_BD2ED0(v48, v3, v45);
      }
      while ( v46 != v47 );
      v46 = v81;
    }
    if ( v46 != (__int64 *)v83 )
      _libc_free((unsigned __int64)v46);
    v49 = sub_A74610(&v71);
    sub_A751C0((__int64)&v81, v61, v49, 3);
    sub_A74A10((__int64)v77, (__int64)&v81);
    sub_29A3490(v85);
    goto LABEL_59;
  }
  return v3;
}
