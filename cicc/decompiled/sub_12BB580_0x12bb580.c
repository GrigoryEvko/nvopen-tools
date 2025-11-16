// Function: sub_12BB580
// Address: 0x12bb580
//
__int64 __fastcall sub_12BB580(__int64 a1, int a2, char *a3)
{
  int v6; // edx
  char *v7; // r12
  size_t v8; // rax
  __int64 v9; // r8
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v14; // r12
  int v15; // ebx
  __int64 v16; // r14
  int v17; // ebx
  int v18; // ebx
  int v19; // ebx
  size_t v20; // rax
  char *v21; // r8
  char *v22; // r8
  size_t v23; // rdx
  int v24; // eax
  int v25; // eax
  char *v26; // [rsp+0h] [rbp-240h]
  size_t n; // [rsp+10h] [rbp-230h]
  char *na; // [rsp+10h] [rbp-230h]
  unsigned __int8 v29; // [rsp+68h] [rbp-1D8h]
  char *v30; // [rsp+68h] [rbp-1D8h]
  char *v31; // [rsp+68h] [rbp-1D8h]
  const char *v32; // [rsp+68h] [rbp-1D8h]
  int v33; // [rsp+74h] [rbp-1CCh] BYREF
  int v34; // [rsp+78h] [rbp-1C8h] BYREF
  int v35; // [rsp+7Ch] [rbp-1C4h] BYREF
  int v36; // [rsp+80h] [rbp-1C0h] BYREF
  int v37; // [rsp+84h] [rbp-1BCh] BYREF
  __int64 v38; // [rsp+88h] [rbp-1B8h] BYREF
  __int64 v39; // [rsp+90h] [rbp-1B0h] BYREF
  char *v40; // [rsp+98h] [rbp-1A8h] BYREF
  char *v41; // [rsp+A0h] [rbp-1A0h] BYREF
  char *s; // [rsp+A8h] [rbp-198h] BYREF
  __int64 v43; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v44; // [rsp+B8h] [rbp-188h] BYREF
  __int64 v45; // [rsp+C0h] [rbp-180h] BYREF
  __int64 v46; // [rsp+C8h] [rbp-178h] BYREF
  __int64 v47; // [rsp+D0h] [rbp-170h] BYREF
  char *v48; // [rsp+D8h] [rbp-168h] BYREF
  __int64 v49; // [rsp+E0h] [rbp-160h] BYREF
  char *v50; // [rsp+E8h] [rbp-158h] BYREF
  char *name; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v52; // [rsp+F8h] [rbp-148h]
  _QWORD v53[2]; // [rsp+100h] [rbp-140h] BYREF
  _BYTE v54[8]; // [rsp+110h] [rbp-130h] BYREF
  unsigned int v55; // [rsp+118h] [rbp-128h]
  _QWORD *v56; // [rsp+120h] [rbp-120h]
  __int64 v57; // [rsp+128h] [rbp-118h]
  _QWORD v58[2]; // [rsp+130h] [rbp-110h] BYREF
  _QWORD *v59; // [rsp+140h] [rbp-100h]
  __int64 v60; // [rsp+148h] [rbp-F8h]
  _QWORD v61[2]; // [rsp+150h] [rbp-F0h] BYREF
  _QWORD *v62; // [rsp+160h] [rbp-E0h]
  __int64 v63; // [rsp+168h] [rbp-D8h]
  _QWORD v64[2]; // [rsp+170h] [rbp-D0h] BYREF
  _QWORD *v65; // [rsp+180h] [rbp-C0h]
  __int64 v66; // [rsp+188h] [rbp-B8h]
  _QWORD v67[2]; // [rsp+190h] [rbp-B0h] BYREF
  _QWORD *v68; // [rsp+1A0h] [rbp-A0h]
  __int64 v69; // [rsp+1A8h] [rbp-98h]
  _QWORD v70[2]; // [rsp+1B0h] [rbp-90h] BYREF
  _QWORD *v71; // [rsp+1C0h] [rbp-80h]
  __int64 v72; // [rsp+1C8h] [rbp-78h]
  _QWORD v73[2]; // [rsp+1D0h] [rbp-70h] BYREF
  __int64 v74; // [rsp+1E0h] [rbp-60h]
  __int64 v75; // [rsp+1E8h] [rbp-58h]
  __int64 v76; // [rsp+1F0h] [rbp-50h]

  v59 = v61;
  v56 = v58;
  v65 = v67;
  v62 = v64;
  v37 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  s = 0;
  v55 = 0;
  v57 = 0;
  LOBYTE(v58[0]) = 0;
  v60 = 0;
  LOBYTE(v61[0]) = 0;
  v63 = 0;
  LOBYTE(v64[0]) = 0;
  v66 = 0;
  v71 = v73;
  v6 = *(_DWORD *)(a1 + 176);
  v68 = v70;
  v76 = 0x1000000000LL;
  LOBYTE(v67[0]) = 0;
  v69 = 0;
  LOBYTE(v70[0]) = 0;
  v72 = 0;
  LOBYTE(v73[0]) = 0;
  v74 = 0;
  v75 = 0;
  if ( (unsigned int)sub_12D2AA0(
                       a2,
                       (_DWORD)a3,
                       v6,
                       (unsigned int)&v33,
                       (unsigned int)&v38,
                       (unsigned int)&v34,
                       (__int64)&v39,
                       (__int64)&v35,
                       (__int64)&v40,
                       (__int64)&v36,
                       (__int64)&v41,
                       (__int64)&v37,
                       (__int64)&s,
                       (__int64)v54) )
  {
    v7 = s;
    if ( s )
    {
      v8 = strlen(s);
      a3 = 0;
      sub_2241130(a1 + 80, 0, *(_QWORD *)(a1 + 88), v7, v8);
      if ( s )
      {
        j_j___libc_free_0_0(s);
        v29 = 0;
        goto LABEL_6;
      }
    }
LABEL_5:
    v29 = 0;
    goto LABEL_6;
  }
  v14 = v38;
  v15 = v33;
  if ( v33 != (_DWORD)v43 || v38 != v44 )
  {
    a3 = (char *)&v44;
    sub_12C7AC0(&v43, &v44);
    LODWORD(v43) = v15;
    v44 = v14;
  }
  v16 = v39;
  v17 = v34;
  if ( v34 != (_DWORD)v45 || v39 != v46 )
  {
    a3 = (char *)&v46;
    sub_12C7AC0(&v45, &v46);
    LODWORD(v45) = v17;
    v46 = v16;
  }
  v18 = v35;
  if ( v35 != (_DWORD)v47 || v40 != v48 )
  {
    v30 = v40;
    a3 = (char *)&v48;
    sub_12C7AC0(&v47, &v48);
    LODWORD(v47) = v18;
    v48 = v30;
  }
  v19 = v36;
  if ( v36 != (_DWORD)v49 || v41 != v50 )
  {
    v31 = v41;
    a3 = (char *)&v50;
    sub_12C7AC0(&v49, &v50);
    LODWORD(v49) = v19;
    v50 = v31;
  }
  if ( (v37 & 0x100) != 0 )
    goto LABEL_5;
  v29 = 1;
  if ( (v37 & 0x200) != 0 )
    goto LABEL_6;
  a3 = &byte_42812CF[-15];
  name = (char *)v53;
  sub_12B9B00((__int64 *)&name, (char)&byte_42812CF[-15], &byte_42812CF[-15], byte_42812CF);
  v32 = getenv(name);
  if ( name != (char *)v53 )
  {
    a3 = (char *)(v53[0] + 1LL);
    j_j___libc_free_0(name, v53[0] + 1LL);
  }
  if ( !v32 )
  {
LABEL_49:
    v29 = sub_12B9F70(v55);
    goto LABEL_6;
  }
  v20 = strlen(v32);
  name = (char *)v53;
  a3 = &byte_42812B2[-6];
  n = v20;
  sub_12B9B00((__int64 *)&name, (char)&byte_42812B2[-6], &byte_42812B2[-6], byte_42812B2);
  v21 = name;
  if ( n == v52 )
  {
    if ( !n || (a3 = name, v26 = name, v25 = memcmp(v32, name, n), v21 = v26, !v25) )
    {
      if ( v21 != (char *)v53 )
      {
        a3 = (char *)(v53[0] + 1LL);
        j_j___libc_free_0(v21, v53[0] + 1LL);
      }
      goto LABEL_5;
    }
  }
  if ( v21 != (char *)v53 )
    j_j___libc_free_0(v21, v53[0] + 1LL);
  name = (char *)v53;
  a3 = &byte_42812AB[-11];
  sub_12B9B00((__int64 *)&name, (char)&byte_42812AB[-11], &byte_42812AB[-11], byte_42812AB);
  v22 = name;
  if ( n != v52 || n && (a3 = name, v23 = n, na = name, v24 = memcmp(v32, name, v23), v22 = na, v24) )
  {
    if ( v22 != (char *)v53 )
    {
      a3 = (char *)(v53[0] + 1LL);
      j_j___libc_free_0(v22, v53[0] + 1LL);
    }
    goto LABEL_49;
  }
  if ( v22 != (char *)v53 )
  {
    a3 = (char *)(v53[0] + 1LL);
    j_j___libc_free_0(v22, v53[0] + 1LL);
  }
  v29 = 1;
LABEL_6:
  v9 = v74;
  if ( HIDWORD(v75) && (_DWORD)v75 )
  {
    v10 = 0;
    v11 = 8LL * (unsigned int)v75;
    do
    {
      v12 = *(_QWORD *)(v9 + v10);
      if ( v12 && v12 != -8 )
      {
        _libc_free(v12, a3);
        v9 = v74;
      }
      v10 += 8;
    }
    while ( v11 != v10 );
  }
  _libc_free(v9, a3);
  if ( v71 != v73 )
    j_j___libc_free_0(v71, v73[0] + 1LL);
  if ( v68 != v70 )
    j_j___libc_free_0(v68, v70[0] + 1LL);
  if ( v65 != v67 )
    j_j___libc_free_0(v65, v67[0] + 1LL);
  if ( v62 != v64 )
    j_j___libc_free_0(v62, v64[0] + 1LL);
  if ( v59 != v61 )
    j_j___libc_free_0(v59, v61[0] + 1LL);
  if ( v56 != v58 )
    j_j___libc_free_0(v56, v58[0] + 1LL);
  sub_12C7AC0(&v49, &v50);
  sub_12C7AC0(&v47, &v48);
  sub_12C7AC0(&v45, &v46);
  sub_12C7AC0(&v43, &v44);
  return v29;
}
