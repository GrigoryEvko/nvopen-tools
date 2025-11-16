// Function: sub_1953CB0
// Address: 0x1953cb0
//
void __fastcall sub_1953CB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 *v10; // rdi
  int v11; // eax
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  unsigned int v14; // ebx
  unsigned __int64 v15; // r14
  int v16; // eax
  int v17; // r8d
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 *v21; // rax
  unsigned __int64 i; // r15
  unsigned __int64 *v23; // r14
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int *v28; // rdi
  unsigned int *v29; // rsi
  __int64 v30; // rbx
  __int64 j; // r15
  int v32; // edx
  int v33; // ecx
  int v34; // r9d
  unsigned int *v35; // r8
  unsigned int v36; // r12d
  __int64 v37; // rax
  unsigned int *v38; // rbx
  unsigned int *v39; // rdx
  unsigned int *v40; // r14
  unsigned __int64 v41; // r12
  __int64 v42; // rax
  unsigned __int64 v43; // rax
  int v44; // r8d
  int v45; // r9d
  __int64 v46; // r15
  unsigned int *v47; // rax
  unsigned int *v48; // rcx
  unsigned __int64 *v49; // r9
  unsigned int v50; // edx
  unsigned __int64 *v51; // rbx
  __int64 v52; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v53; // [rsp+18h] [rbp-D8h]
  int v54; // [rsp+18h] [rbp-D8h]
  int v55; // [rsp+20h] [rbp-D0h]
  __int64 v57; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v58; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v59; // [rsp+48h] [rbp-A8h] BYREF
  unsigned int *v60; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v61; // [rsp+58h] [rbp-98h]
  _BYTE v62[16]; // [rsp+60h] [rbp-90h] BYREF
  unsigned int *v63; // [rsp+70h] [rbp-80h] BYREF
  __int64 v64; // [rsp+78h] [rbp-78h]
  _BYTE v65[16]; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 *v66; // [rsp+90h] [rbp-60h] BYREF
  __int64 v67; // [rsp+98h] [rbp-58h]
  _BYTE v68[80]; // [rsp+A0h] [rbp-50h] BYREF

  if ( !*(_BYTE *)(a1 + 48) )
    return;
  v9 = sub_1368AA0(*(__int64 **)(a1 + 32), a3);
  v10 = *(__int64 **)(a1 + 32);
  v57 = v9;
  v53 = sub_1368AA0(v10, a4);
  v11 = sub_13774B0(*(_QWORD *)(a1 + 40), a3, a5);
  v58 = sub_16AF500(&v57, v11);
  v12 = sub_16AF5D0(&v57, v53);
  sub_136C010(*(__int64 **)(a1 + 32), a3, v12);
  v66 = (unsigned __int64 *)v68;
  v67 = 0x400000000LL;
  v13 = sub_157EBA0(a3);
  if ( !v13 )
  {
    v51 = (unsigned __int64 *)v68;
    v50 = 0;
    goto LABEL_44;
  }
  v14 = 0;
  v55 = sub_15F4D60(v13);
  v15 = sub_157EBA0(a3);
  if ( !v55 )
  {
    v49 = v66;
    v50 = v67;
    v51 = &v66[(unsigned int)v67];
    goto LABEL_12;
  }
  do
  {
    v20 = sub_15F4DF0(v15, v14);
    if ( a5 == v20 )
    {
      v18 = sub_16AF5D0(&v58, v53);
      v19 = (unsigned int)v67;
      if ( (unsigned int)v67 < HIDWORD(v67) )
        goto LABEL_7;
    }
    else
    {
      v16 = sub_13774B0(*(_QWORD *)(a1 + 40), a3, v20);
      v18 = sub_16AF500(&v57, v16);
      v19 = (unsigned int)v67;
      if ( (unsigned int)v67 < HIDWORD(v67) )
        goto LABEL_7;
    }
    v52 = v18;
    sub_16CD150((__int64)&v66, v68, 0, 8, v17, v18);
    v19 = (unsigned int)v67;
    v18 = v52;
LABEL_7:
    ++v14;
    v66[v19] = v18;
    v50 = v67 + 1;
    LODWORD(v67) = v67 + 1;
  }
  while ( v55 != v14 );
  v49 = v66;
  v51 = &v66[v50];
LABEL_12:
  if ( v49 == v51 )
  {
LABEL_44:
    v43 = *v51;
    v60 = (unsigned int *)v62;
    v61 = 0x400000000LL;
    if ( v43 )
    {
      v28 = (unsigned int *)v62;
      v29 = (unsigned int *)v62;
      goto LABEL_23;
    }
    goto LABEL_45;
  }
  v21 = v49 + 1;
  for ( i = *v49; v21 != v51; ++v21 )
  {
    if ( i < *v21 )
      i = *v21;
  }
  v60 = (unsigned int *)v62;
  v61 = 0x400000000LL;
  if ( i )
  {
    v23 = v49;
    do
    {
      v25 = sub_16AF730(*v23, i);
      v26 = (unsigned int)v61;
      if ( (unsigned int)v61 >= HIDWORD(v61) )
      {
        v54 = v25;
        sub_16CD150((__int64)&v60, v62, 0, 4, v24, v25);
        v26 = (unsigned int)v61;
        v25 = v54;
      }
      ++v23;
      v60[v26] = v25;
      v27 = (unsigned int)(v61 + 1);
      LODWORD(v61) = v61 + 1;
    }
    while ( v51 != v23 );
    v28 = v60;
    v29 = &v60[v27];
LABEL_23:
    sub_1953BB0(v28, v29);
LABEL_24:
    v30 = (unsigned int)v61;
    goto LABEL_25;
  }
LABEL_45:
  sub_16AF710(&v63, 1u, v50);
  v46 = (unsigned int)v67;
  LODWORD(v61) = 0;
  v30 = (unsigned int)v67;
  if ( HIDWORD(v61) < (unsigned int)v67 )
    sub_16CD150((__int64)&v60, v62, (unsigned int)v67, 4, v44, v45);
  v47 = v60;
  LODWORD(v61) = v30;
  v48 = &v60[v46];
  if ( v60 != v48 )
  {
    do
    {
      if ( v47 )
        *v47 = (unsigned int)v63;
      ++v47;
    }
    while ( v48 != v47 );
    goto LABEL_24;
  }
LABEL_25:
  if ( (int)v30 > 0 )
  {
    for ( j = 0; j != v30; ++j )
    {
      v32 = j;
      v33 = v60[j];
      sub_1379150(*(_QWORD *)(a1 + 40), a3, v32, v33);
    }
    LODWORD(v30) = v61;
  }
  if ( (unsigned int)v30 > 1 && sub_1953390(a1, a3) )
  {
    v64 = 0x400000000LL;
    v63 = (unsigned int *)v65;
    v35 = &v60[(unsigned int)v61];
    if ( v60 != v35 )
    {
      v36 = *v60;
      v37 = 0;
      v38 = v60 + 1;
      v39 = (unsigned int *)v65;
      v40 = &v60[(unsigned int)v61];
      while ( 1 )
      {
        v39[v37] = v36;
        v37 = (unsigned int)(v64 + 1);
        LODWORD(v64) = v64 + 1;
        if ( v40 == v38 )
          break;
        v36 = *v38;
        if ( HIDWORD(v64) <= (unsigned int)v37 )
        {
          sub_16CD150((__int64)&v63, v65, 0, 4, (int)v35, v34);
          v37 = (unsigned int)v64;
        }
        v39 = v63;
        ++v38;
      }
    }
    v41 = sub_157EBA0(a3);
    v59 = sub_157E9C0(*(_QWORD *)(v41 + 40));
    v42 = sub_161BD30(&v59, v63, (unsigned int)v64);
    sub_1625C10(v41, 2, v42);
    if ( v63 != (unsigned int *)v65 )
      _libc_free((unsigned __int64)v63);
  }
  if ( v60 != (unsigned int *)v62 )
    _libc_free((unsigned __int64)v60);
  if ( v66 != (unsigned __int64 *)v68 )
    _libc_free((unsigned __int64)v66);
}
