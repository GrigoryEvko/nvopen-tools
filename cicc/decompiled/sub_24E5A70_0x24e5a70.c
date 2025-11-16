// Function: sub_24E5A70
// Address: 0x24e5a70
//
void __fastcall sub_24E5A70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  unsigned __int64 *v7; // rax
  _BOOL4 v8; // r15d
  __int64 *v9; // r14
  __int64 v10; // rsi
  __int64 *v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rax
  char *v14; // rbx
  char *v15; // r15
  _BYTE *v16; // rax
  _QWORD *v17; // r14
  __int64 v18; // rdi
  __int64 *v19; // rbx
  __int64 *v20; // r15
  _BYTE *v21; // rax
  _QWORD *v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdi
  _BYTE *v25; // rbx
  _BYTE *v26; // r12
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  int v30; // r9d
  __int64 v31; // r10
  __int64 v32; // rdx
  __int64 v33; // rdi
  char v34; // al
  __int64 v35; // rax
  int v36; // r9d
  __int64 v37; // r10
  __int64 v38; // rdx
  __int64 v39; // rdi
  char v40; // al
  int v41; // [rsp+10h] [rbp-1A0h]
  __int64 v42; // [rsp+10h] [rbp-1A0h]
  __int64 *v43; // [rsp+18h] [rbp-198h]
  __int64 *v44; // [rsp+18h] [rbp-198h]
  __int64 v45; // [rsp+18h] [rbp-198h]
  int v46; // [rsp+18h] [rbp-198h]
  __int64 v47; // [rsp+20h] [rbp-190h] BYREF
  __int64 v48; // [rsp+28h] [rbp-188h]
  __int64 v49; // [rsp+30h] [rbp-180h] BYREF
  unsigned int v50; // [rsp+38h] [rbp-178h]
  unsigned __int64 v51[2]; // [rsp+70h] [rbp-140h] BYREF
  char v52; // [rsp+80h] [rbp-130h] BYREF
  _BYTE *v53; // [rsp+88h] [rbp-128h]
  __int64 v54; // [rsp+90h] [rbp-120h]
  _BYTE v55[56]; // [rsp+98h] [rbp-118h] BYREF
  __int64 v56; // [rsp+D0h] [rbp-E0h]
  __int64 v57; // [rsp+D8h] [rbp-D8h]
  char v58; // [rsp+E0h] [rbp-D0h]
  __int64 v59; // [rsp+E4h] [rbp-CCh]
  char *v60; // [rsp+F0h] [rbp-C0h] BYREF
  int v61; // [rsp+F8h] [rbp-B8h]
  char v62; // [rsp+100h] [rbp-B0h] BYREF
  __int64 *v63; // [rsp+140h] [rbp-70h]
  int v64; // [rsp+148h] [rbp-68h]
  char v65; // [rsp+150h] [rbp-60h] BYREF

  sub_24E4250(&v60, *(_QWORD *)(a1 + 280), a3, a4, a5, a6);
  v7 = (unsigned __int64 *)&v49;
  v47 = 0;
  v48 = 1;
  do
  {
    *v7 = -4096;
    v7 += 2;
  }
  while ( v7 != v51 );
  v8 = sub_CC7F40(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL) + 232LL);
  v9 = (__int64 *)v60;
  v43 = (__int64 *)&v60[8 * v61];
  if ( v60 != (char *)v43 )
  {
    do
    {
      v10 = *v9++;
      sub_31649C0(&v47, v10, v8);
    }
    while ( v43 != v9 );
  }
  v44 = &v63[v64];
  if ( v44 != v63 )
  {
    v11 = v63;
    do
    {
      v12 = *v11++;
      sub_3164D90(&v47, v12, v8);
    }
    while ( v44 != v11 );
  }
  v13 = *(_QWORD *)(a1 + 280);
  v59 = 0;
  v51[0] = (unsigned __int64)&v52;
  v51[1] = 0x100000000LL;
  v56 = 0;
  v58 = 0;
  v57 = v13;
  LODWORD(v13) = *(_DWORD *)(v13 + 92);
  v53 = v55;
  HIDWORD(v59) = v13;
  v54 = 0x600000000LL;
  sub_B1F440((__int64)v51);
  v14 = v60;
  v15 = &v60[8 * v61];
  if ( v60 != v15 )
  {
    do
    {
      while ( 1 )
      {
        v17 = *(_QWORD **)v14;
        v18 = *(_QWORD *)(*(_QWORD *)(a1 + 280) + 80LL);
        if ( v18 )
          v18 -= 24;
        if ( !(unsigned __int8)sub_D0E9D0(v18, *(_QWORD *)(*(_QWORD *)v14 + 40LL), 0, (__int64)v51, 0) )
          break;
        v16 = (_BYTE *)sub_B58EB0((__int64)v17, 0);
        if ( v16 && *v16 == 60 )
        {
          v29 = sub_B58EB0((__int64)v17, 0);
          v30 = 0;
          v31 = *(_QWORD *)(v29 + 16);
          if ( !v31 )
            break;
          do
          {
            v32 = *(_QWORD *)(v31 + 24);
            if ( *(_BYTE *)v32 > 0x1Cu && *(_BYTE *)v32 != 60 )
            {
              v41 = v30;
              v45 = v31;
              v33 = *(_QWORD *)(*(_QWORD *)(a1 + 280) + 80LL);
              if ( v33 )
                v33 -= 24;
              v34 = sub_D0E9D0(v33, *(_QWORD *)(v32 + 40), 0, (__int64)v51, 0);
              v31 = v45;
              v30 = v41 - ((v34 == 0) - 1);
            }
            v31 = *(_QWORD *)(v31 + 8);
          }
          while ( v31 );
          if ( !v30 )
            break;
        }
        v14 += 8;
        if ( v15 == v14 )
          goto LABEL_17;
      }
      sub_B43D60(v17);
      v14 += 8;
    }
    while ( v15 != v14 );
  }
LABEL_17:
  v19 = v63;
  v20 = &v63[v64];
  if ( v63 != v20 )
  {
    do
    {
      while ( 1 )
      {
        v22 = (_QWORD *)*v19;
        v23 = sub_B140C0(*v19);
        v24 = *(_QWORD *)(*(_QWORD *)(a1 + 280) + 80LL);
        if ( v24 )
          v24 -= 24;
        if ( !(unsigned __int8)sub_D0E9D0(v24, v23, 0, (__int64)v51, 0) )
          break;
        v21 = (_BYTE *)sub_B12A50((__int64)v22, 0);
        if ( v21 && *v21 == 60 )
        {
          v35 = sub_B12A50((__int64)v22, 0);
          v36 = 0;
          v37 = *(_QWORD *)(v35 + 16);
          if ( !v37 )
            break;
          do
          {
            v38 = *(_QWORD *)(v37 + 24);
            if ( *(_BYTE *)v38 != 60 && *(_BYTE *)v38 > 0x1Cu )
            {
              v42 = v37;
              v46 = v36;
              v39 = *(_QWORD *)(*(_QWORD *)(a1 + 280) + 80LL);
              if ( v39 )
                v39 -= 24;
              v40 = sub_D0E9D0(v39, *(_QWORD *)(v38 + 40), 0, (__int64)v51, 0);
              v37 = v42;
              v36 = v46 - ((v40 == 0) - 1);
            }
            v37 = *(_QWORD *)(v37 + 8);
          }
          while ( v37 );
          if ( !v36 )
            break;
        }
        if ( v20 == ++v19 )
          goto LABEL_26;
      }
      sub_B14290(v22);
      ++v19;
    }
    while ( v20 != v19 );
  }
LABEL_26:
  v25 = v53;
  v26 = &v53[8 * (unsigned int)v54];
  if ( v53 != v26 )
  {
    do
    {
      v27 = *((_QWORD *)v26 - 1);
      v26 -= 8;
      if ( v27 )
      {
        v28 = *(_QWORD *)(v27 + 24);
        if ( v28 != v27 + 40 )
          _libc_free(v28);
        j_j___libc_free_0(v27);
      }
    }
    while ( v25 != v26 );
    v26 = v53;
  }
  if ( v26 != v55 )
    _libc_free((unsigned __int64)v26);
  if ( (char *)v51[0] != &v52 )
    _libc_free(v51[0]);
  if ( (v48 & 1) == 0 )
    sub_C7D6A0(v49, 16LL * v50, 8);
  if ( v63 != (__int64 *)&v65 )
    _libc_free((unsigned __int64)v63);
  if ( v60 != &v62 )
    _libc_free((unsigned __int64)v60);
}
