// Function: sub_2233F50
// Address: 0x2233f50
//
_QWORD *__fastcall sub_2233F50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        _DWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        _DWORD *a10)
{
  __int64 *v10; // r12
  void *v11; // rsp
  char v13; // dl
  __int64 v14; // r15
  unsigned __int64 v15; // r13
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r12
  __int64 v19; // rsi
  size_t v20; // r15
  __int64 v21; // rax
  size_t v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  char v26; // r9
  char v27; // si
  __int64 v28; // r8
  unsigned __int64 i; // rsi
  char v30; // r12
  _BYTE *v31; // rax
  size_t v32; // r15
  const char *v33; // rsi
  size_t v34; // r15
  size_t v35; // rax
  __int64 v36; // rsi
  size_t v37; // r12
  char v38; // bl
  char v39; // r13
  int v40; // eax
  char v41; // bl
  _QWORD *v42; // rdi
  _BYTE *v43; // rax
  char v44; // r14
  __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rax
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  int v52; // eax
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // [rsp+0h] [rbp-70h]
  unsigned __int8 v56; // [rsp+6h] [rbp-6Ah]
  char v57; // [rsp+7h] [rbp-69h]
  _DWORD *v58; // [rsp+8h] [rbp-68h]
  size_t v59; // [rsp+10h] [rbp-60h]
  __int64 v60; // [rsp+18h] [rbp-58h]
  __int64 *v61; // [rsp+20h] [rbp-50h] BYREF
  __int64 v62; // [rsp+28h] [rbp-48h]
  _QWORD *v63; // [rsp+30h] [rbp-40h] BYREF
  __int64 v64; // [rsp+38h] [rbp-38h]

  v63 = (_QWORD *)a2;
  v61 = a4;
  v64 = a3;
  v62 = a5;
  v58 = a6;
  v10 = (__int64 *)sub_222F790((_QWORD *)(a9 + 208), a2);
  v11 = alloca(4 * a8 + 8);
  if ( sub_2233E50((__int64)&v63, (__int64)&v61) )
    goto LABEL_2;
  v13 = sub_2233F00((__int64)&v63);
  if ( !a8 )
    goto LABEL_2;
  v14 = 0;
  v15 = 0;
  do
  {
    while ( **(_BYTE **)(a7 + 8 * v14) != v13 )
    {
      v16 = *v10;
      LOBYTE(v60) = v13;
      v17 = (*(__int64 (__fastcall **)(__int64 *))(v16 + 16))(v10);
      v13 = v60;
      if ( (_BYTE)v60 == v17 )
        break;
      if ( a8 == ++v14 )
        goto LABEL_10;
    }
    *(&v55 + v15++) = v14++;
  }
  while ( a8 != v14 );
LABEL_10:
  v59 = 0;
  if ( v15 <= 1 )
    goto LABEL_30;
  do
  {
    v18 = v55;
    v19 = 1;
    v20 = strlen(*(const char **)(a7 + 8LL * v55));
    do
    {
      v21 = *(&v55 + v19);
      v60 = v19;
      v22 = strlen(*(const char **)(a7 + 8 * v21));
      if ( v20 > v22 )
        v20 = v22;
      v19 = v60 + 1;
    }
    while ( v60 + 1 < v15 );
    v24 = v63[2];
    if ( v24 >= v63[3] )
      (*(void (__fastcall **)(_QWORD *))(*v63 + 80LL))(v63);
    else
      v63[2] = v24 + 1;
    LODWORD(v25) = ++v59;
    LODWORD(v64) = -1;
    if ( v59 >= v20 )
      goto LABEL_2;
    v26 = 1;
    if ( v63 )
    {
      v25 = v63[3];
      v26 = 0;
      if ( v63[2] >= v25 )
      {
        v50 = *v63;
        LOBYTE(v60) = 0;
        LODWORD(v25) = (*(__int64 (**)(void))(v50 + 72))();
        v26 = v60;
        if ( (_DWORD)v25 == -1 )
        {
          v63 = 0;
          v26 = 1;
        }
      }
    }
    v27 = (_DWORD)v62 == -1;
    LOBYTE(v25) = v27 & (v61 != 0);
    v28 = (unsigned int)v25;
    if ( (_BYTE)v25 )
    {
      v27 = 0;
      if ( v61[2] >= (unsigned __int64)v61[3] )
      {
        v47 = *v61;
        v56 = v28;
        v57 = v26;
        LOBYTE(v60) = 0;
        v48 = (*(__int64 (**)(void))(v47 + 72))();
        v27 = v60;
        v26 = v57;
        v28 = v56;
        if ( v48 == -1 )
        {
          v61 = 0;
          v27 = v56;
        }
      }
    }
    if ( v27 == v26 )
      goto LABEL_2;
    for ( i = 0; ; v18 = *(&v55 + i) )
    {
      v30 = *(_BYTE *)(*(_QWORD *)(a7 + 8 * v18) + v59);
      LOBYTE(v31) = v64;
      if ( v63 && (_DWORD)v64 == -1 )
      {
        v31 = (_BYTE *)v63[2];
        if ( (unsigned __int64)v31 >= v63[3] )
        {
          v49 = *v63;
          v60 = i;
          LODWORD(v31) = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64, __int64, size_t, __int64))(v49 + 72))(
                           v63,
                           i,
                           v23,
                           v59,
                           v28);
          i = v60;
          if ( (_DWORD)v31 == -1 )
            v63 = 0;
        }
        else
        {
          LOBYTE(v31) = *v31;
        }
      }
      if ( v30 == (_BYTE)v31 )
        break;
      --v15;
      *(&v55 + i) = *(&v55 + v15);
      if ( v15 <= i )
        goto LABEL_29;
LABEL_24:
      ;
    }
    if ( v15 > ++i )
      goto LABEL_24;
LABEL_29:
    ;
  }
  while ( v15 > 1 );
LABEL_30:
  if ( v15 != 1 )
    goto LABEL_2;
  sub_22408B0(v63);
  v32 = v59;
  LODWORD(v64) = -1;
  v33 = *(const char **)(a7 + 8LL * v55);
  LODWORD(v59) = v55;
  v34 = v32 + 1;
  v60 = (__int64)v33;
  v35 = strlen(v33);
  v36 = v60;
  v37 = v35;
  if ( v34 < v35 )
  {
    while ( 1 )
    {
      v44 = 1;
      if ( v63 )
      {
        v44 = 0;
        if ( v63[2] >= v63[3] )
        {
          v45 = *v63;
          v60 = v36;
          v46 = (*(__int64 (**)(void))(v45 + 72))();
          v36 = v60;
          if ( v46 == -1 )
          {
            v63 = 0;
            v44 = 1;
          }
        }
      }
      v38 = (_DWORD)v62 == -1;
      v39 = v38 & (v61 != 0);
      if ( v39 )
      {
        v38 = 0;
        if ( v61[2] >= (unsigned __int64)v61[3] )
        {
          v51 = *v61;
          v60 = v36;
          v52 = (*(__int64 (**)(void))(v51 + 72))();
          v36 = v60;
          if ( v52 == -1 )
          {
            v61 = 0;
            v38 = v39;
          }
        }
      }
      if ( v44 == v38 )
        goto LABEL_2;
      LOBYTE(v40) = v64;
      v41 = *(_BYTE *)(v36 + v34);
      v42 = v63;
      if ( (_DWORD)v64 == -1 && v63 )
      {
        v43 = (_BYTE *)v63[2];
        if ( (unsigned __int64)v43 < v63[3] )
        {
          if ( v41 != *v43 )
            goto LABEL_2;
          goto LABEL_39;
        }
        v54 = *v63;
        v60 = v36;
        v40 = (*(__int64 (**)(void))(v54 + 72))();
        v36 = v60;
        if ( v40 == -1 )
          v63 = 0;
      }
      if ( v41 != (_BYTE)v40 )
        goto LABEL_2;
      v42 = v63;
      v43 = (_BYTE *)v63[2];
      if ( (unsigned __int64)v43 >= v63[3] )
      {
        v53 = *v63;
        v60 = v36;
        (*(void (__fastcall **)(_QWORD *))(v53 + 80))(v63);
        v36 = v60;
        goto LABEL_40;
      }
LABEL_39:
      v42[2] = v43 + 1;
LABEL_40:
      ++v34;
      LODWORD(v64) = -1;
      if ( v34 == v37 )
        goto LABEL_63;
    }
  }
  if ( v34 == v35 )
  {
LABEL_63:
    *v58 = v59;
    return v63;
  }
LABEL_2:
  *a10 |= 4u;
  return v63;
}
