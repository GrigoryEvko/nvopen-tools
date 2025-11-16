// Function: sub_2580AE0
// Address: 0x2580ae0
//
__int64 __fastcall sub_2580AE0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r15
  __m128i v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // r12d
  _BYTE *v10; // rbx
  unsigned __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // r13
  _BYTE *v15; // rbx
  unsigned __int64 v16; // r13
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 v19; // r13
  char v21; // r15
  __int64 v22; // rdx
  _BYTE *v23; // rcx
  int v24; // r14d
  __int64 v25; // r13
  __int64 v26; // r12
  __int64 v27; // rbx
  int v28; // ebx
  _QWORD *v29; // r14
  int v30; // eax
  _QWORD *v31; // r13
  char v32; // al
  __int64 v33; // r15
  char v34; // bl
  char v35; // r14
  char v36; // al
  char v37; // r13
  __int64 v38; // [rsp+8h] [rbp-2F8h]
  _BYTE *v39; // [rsp+10h] [rbp-2F0h]
  __int64 v40; // [rsp+10h] [rbp-2F0h]
  const void ***v41; // [rsp+30h] [rbp-2D0h]
  _QWORD *v42; // [rsp+40h] [rbp-2C0h]
  _QWORD *v43; // [rsp+40h] [rbp-2C0h]
  _BYTE *v44; // [rsp+40h] [rbp-2C0h]
  char v45; // [rsp+6Eh] [rbp-292h] BYREF
  char v46; // [rsp+6Fh] [rbp-291h] BYREF
  __int64 v47; // [rsp+70h] [rbp-290h] BYREF
  unsigned int v48; // [rsp+78h] [rbp-288h]
  __m128i v49; // [rsp+80h] [rbp-280h] BYREF
  __m128i v50; // [rsp+90h] [rbp-270h] BYREF
  __int64 v51; // [rsp+A0h] [rbp-260h] BYREF
  __int64 v52; // [rsp+A8h] [rbp-258h]
  __int64 v53; // [rsp+B0h] [rbp-250h]
  __int64 v54; // [rsp+B8h] [rbp-248h]
  _BYTE *v55; // [rsp+C0h] [rbp-240h]
  __int64 v56; // [rsp+C8h] [rbp-238h]
  _BYTE v57[128]; // [rsp+D0h] [rbp-230h] BYREF
  __int64 v58; // [rsp+150h] [rbp-1B0h] BYREF
  __int64 v59; // [rsp+158h] [rbp-1A8h]
  __int64 v60; // [rsp+160h] [rbp-1A0h]
  __int64 v61; // [rsp+168h] [rbp-198h]
  _BYTE *v62; // [rsp+170h] [rbp-190h]
  __int64 v63; // [rsp+178h] [rbp-188h]
  _BYTE v64[128]; // [rsp+180h] [rbp-180h] BYREF
  _BYTE v65[256]; // [rsp+200h] [rbp-100h] BYREF

  v41 = (const void ***)(a1 + 88);
  sub_2560F70((__int64)v65, a1 + 88);
  v5 = *(_QWORD *)(a3 - 64);
  v6 = *(_QWORD *)(a3 - 32);
  v55 = v57;
  v62 = v64;
  v45 = 0;
  v46 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v56 = 0x800000000LL;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v63 = 0x800000000LL;
  v7.m128i_i64[0] = sub_250D2C0(v5, 0);
  v49 = v7;
  if ( !(unsigned __int8)sub_2580850(a1, a2, &v49, &v51, &v45, 0)
    || (v50.m128i_i64[0] = sub_250D2C0(v6, 0),
        v50.m128i_i64[1] = v8,
        !(unsigned __int8)sub_2580850(a1, a2, &v50, &v58, &v46, 0)) )
  {
    v9 = 0;
    *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
    goto LABEL_4;
  }
  v48 = *(_DWORD *)(*(_QWORD *)(v6 + 8) + 8LL) >> 8;
  if ( v48 > 0x40 )
    sub_C43690((__int64)&v47, 0, 0);
  else
    v47 = 0;
  v21 = v46;
  if ( v45 )
  {
    if ( v46 )
    {
      *(_BYTE *)(a1 + 288) = *(_DWORD *)(a1 + 152) == 0;
    }
    else
    {
      v43 = &v62[16 * (unsigned int)v63];
      if ( v43 != (_QWORD *)v62 )
      {
        v40 = a1;
        LOBYTE(v24) = 0;
        v31 = v62;
        while ( 1 )
        {
          v32 = sub_B532C0((__int64)&v47, v31, *(_WORD *)(a3 + 2) & 0x3F);
          LOBYTE(v24) = v32 ^ 1 | v24;
          v21 |= v32;
          if ( v21 )
          {
            if ( (_BYTE)v24 )
              goto LABEL_53;
          }
          v31 += 2;
          if ( v43 == v31 )
            goto LABEL_66;
        }
      }
    }
LABEL_55:
    v9 = (unsigned __int8)sub_255BE50((__int64)v65, v41);
    goto LABEL_56;
  }
  v22 = 16LL * (unsigned int)v56;
  v23 = &v55[v22];
  if ( !v46 )
  {
    v39 = &v55[v22];
    if ( v23 != v55 )
    {
      v38 = a1;
      v24 = 0;
      v25 = a3;
      v26 = (__int64)v55;
      do
      {
        v27 = 16LL * (unsigned int)v63;
        v42 = &v62[v27];
        if ( &v62[v27] != v62 )
        {
          v28 = v24;
          v29 = v62;
          do
          {
            LOBYTE(v30) = sub_B532C0(v26, v29, *(_WORD *)(v25 + 2) & 0x3F);
            v28 |= v30 ^ 1;
            v21 |= v30;
            if ( v21 && (_BYTE)v28 )
            {
              v9 = 0;
              *(_BYTE *)(v38 + 105) = *(_BYTE *)(v38 + 104);
              goto LABEL_56;
            }
            v29 += 2;
          }
          while ( v42 != v29 );
          v24 = v28;
        }
        v26 += 16;
      }
      while ( v39 != (_BYTE *)v26 );
LABEL_66:
      v37 = v24;
LABEL_67:
      if ( v21 )
      {
        v50.m128i_i32[2] = 1;
        v50.m128i_i64[0] = 1;
        sub_25761A0((__int64)v41, (const void **)&v50);
        sub_969240(v50.m128i_i64);
      }
      if ( v37 )
      {
        v50.m128i_i32[2] = 1;
        v50.m128i_i64[0] = 0;
        sub_25761A0((__int64)v41, (const void **)&v50);
        sub_969240(v50.m128i_i64);
      }
    }
    goto LABEL_55;
  }
  v44 = &v55[v22];
  if ( v23 == v55 )
    goto LABEL_55;
  v40 = a1;
  v33 = (__int64)v55;
  v34 = v45;
  v35 = 0;
  while ( 1 )
  {
    v36 = sub_B532C0(v33, &v47, *(_WORD *)(a3 + 2) & 0x3F);
    v35 |= v36 ^ 1;
    v34 |= v36;
    if ( v34 )
    {
      if ( v35 )
        break;
    }
    v33 += 16;
    if ( v44 == (_BYTE *)v33 )
    {
      v37 = v35;
      v21 = v34;
      goto LABEL_67;
    }
  }
LABEL_53:
  v9 = 0;
  *(_BYTE *)(v40 + 105) = *(_BYTE *)(v40 + 104);
LABEL_56:
  sub_969240(&v47);
LABEL_4:
  v10 = v62;
  v11 = (unsigned __int64)&v62[16 * (unsigned int)v63];
  if ( v62 != (_BYTE *)v11 )
  {
    do
    {
      v11 -= 16LL;
      if ( *(_DWORD *)(v11 + 8) > 0x40u && *(_QWORD *)v11 )
        j_j___libc_free_0_0(*(_QWORD *)v11);
    }
    while ( v10 != (_BYTE *)v11 );
    v11 = (unsigned __int64)v62;
  }
  if ( (_BYTE *)v11 != v64 )
    _libc_free(v11);
  v12 = (unsigned int)v61;
  if ( (_DWORD)v61 )
  {
    v13 = v59;
    v49.m128i_i32[2] = 0;
    v49.m128i_i64[0] = -1;
    v50.m128i_i32[2] = 0;
    v14 = v59 + 16LL * (unsigned int)v61;
    v50.m128i_i64[0] = -2;
    do
    {
      if ( *(_DWORD *)(v13 + 8) > 0x40u && *(_QWORD *)v13 )
        j_j___libc_free_0_0(*(_QWORD *)v13);
      v13 += 16;
    }
    while ( v14 != v13 );
    sub_969240(v50.m128i_i64);
    sub_969240(v49.m128i_i64);
    v12 = (unsigned int)v61;
  }
  sub_C7D6A0(v59, 16 * v12, 8);
  v15 = v55;
  v16 = (unsigned __int64)&v55[16 * (unsigned int)v56];
  if ( v55 != (_BYTE *)v16 )
  {
    do
    {
      v16 -= 16LL;
      if ( *(_DWORD *)(v16 + 8) > 0x40u && *(_QWORD *)v16 )
        j_j___libc_free_0_0(*(_QWORD *)v16);
    }
    while ( v15 != (_BYTE *)v16 );
    v16 = (unsigned __int64)v55;
  }
  if ( (_BYTE *)v16 != v57 )
    _libc_free(v16);
  v17 = (unsigned int)v54;
  if ( (_DWORD)v54 )
  {
    v18 = v52;
    v50.m128i_i32[2] = 0;
    v50.m128i_i64[0] = -1;
    LODWORD(v59) = 0;
    v19 = v52 + 16LL * (unsigned int)v54;
    v58 = -2;
    do
    {
      if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
        j_j___libc_free_0_0(*(_QWORD *)v18);
      v18 += 16;
    }
    while ( v19 != v18 );
    sub_969240(&v58);
    sub_969240(v50.m128i_i64);
    v17 = (unsigned int)v54;
  }
  sub_C7D6A0(v52, 16 * v17, 8);
  sub_25485A0((__int64)v65);
  return v9;
}
