// Function: sub_12E86C0
// Address: 0x12e86c0
//
__int64 __fastcall sub_12E86C0(__int64 a1, unsigned int a2, __int64 a3, _QWORD *a4)
{
  _DWORD *v7; // rax
  __int64 v8; // r12
  _QWORD *v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // r11
  __int64 v12; // rax
  __int64 (*v13)(void); // rax
  _QWORD *v14; // rdi
  _QWORD *v15; // rax
  _QWORD *i; // rsi
  bool v17; // zf
  __int64 v18; // rdx
  __int64 v19; // rsi
  _QWORD *v20; // r12
  __int64 result; // rax
  _BYTE *v22; // r12
  _BYTE *v23; // rbx
  _BYTE *v24; // rdi
  __int64 v25; // rsi
  _QWORD *v26; // rax
  char v27; // cl
  char v28; // dl
  _QWORD *v29; // rdx
  _QWORD *v30; // r12
  _QWORD *v31; // r12
  _BYTE *v32; // rbx
  _BYTE *v33; // r12
  _BYTE *v34; // rdi
  _DWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  unsigned int v38; // eax
  __int64 v39; // rdi
  char *v40; // rax
  __int64 v41; // rdx
  int v42; // [rsp+18h] [rbp-1E8h]
  _QWORD *v43; // [rsp+28h] [rbp-1D8h] BYREF
  _QWORD *v44; // [rsp+30h] [rbp-1D0h] BYREF
  __int64 v45; // [rsp+38h] [rbp-1C8h] BYREF
  __int64 *v46; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v47; // [rsp+48h] [rbp-1B8h]
  char *v48; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 v49; // [rsp+58h] [rbp-1A8h]
  _QWORD v50[2]; // [rsp+60h] [rbp-1A0h] BYREF
  _QWORD *v51; // [rsp+70h] [rbp-190h]
  __int64 v52; // [rsp+78h] [rbp-188h]
  _QWORD v53[3]; // [rsp+80h] [rbp-180h] BYREF
  int v54; // [rsp+98h] [rbp-168h]
  _QWORD *v55; // [rsp+A0h] [rbp-160h]
  __int64 v56; // [rsp+A8h] [rbp-158h]
  _QWORD v57[2]; // [rsp+B0h] [rbp-150h] BYREF
  _QWORD *v58; // [rsp+C0h] [rbp-140h]
  __int64 v59; // [rsp+C8h] [rbp-138h]
  _QWORD v60[2]; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v61; // [rsp+E0h] [rbp-120h]
  __int64 v62; // [rsp+E8h] [rbp-118h]
  __int64 v63; // [rsp+F0h] [rbp-110h]
  _BYTE *v64; // [rsp+F8h] [rbp-108h]
  __int64 v65; // [rsp+100h] [rbp-100h]
  _BYTE v66[248]; // [rsp+108h] [rbp-F8h] BYREF

  if ( **(_QWORD **)a1 && (unsigned int)sub_16827A0() )
    goto LABEL_71;
  v7 = (_DWORD *)sub_1C42D70(4, 4);
  *v7 = 2;
  sub_16D40E0(qword_4FBB3B0, v7);
  sub_16C2450(&v43, *a4, a4[1], byte_3F871B3, 0, 0);
  v52 = 0;
  v51 = v53;
  v55 = v57;
  v8 = 8LL * a2;
  v65 = 0x400000000LL;
  v9 = *(_QWORD **)(a1 + 8);
  LOBYTE(v53[0]) = 0;
  v53[2] = 0;
  v54 = 0;
  v56 = 0;
  LOBYTE(v57[0]) = 0;
  v58 = v60;
  v59 = 0;
  LOBYTE(v60[0]) = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = v66;
  v50[1] = 0;
  v10 = (__int64 *)v43[1];
  v50[0] = 0;
  v11 = *(_QWORD *)(*v9 + v8);
  v12 = v43[2];
  v46 = v10;
  v47 = v12 - (_QWORD)v10;
  v13 = *(__int64 (**)(void))(*v43 + 16LL);
  if ( (char *)v13 == (char *)sub_12BCB10 )
  {
    v49 = 14;
    v48 = "Unknown buffer";
  }
  else
  {
    v42 = v11;
    v40 = (char *)v13();
    LODWORD(v11) = v42;
    v48 = v40;
    v49 = v41;
  }
  sub_166F050(
    (unsigned int)&v44,
    (unsigned int)v50,
    v11,
    1,
    (unsigned int)byte_3F871B3,
    0,
    (__int64)v46,
    v47,
    (__int64)v48,
    v49);
  v14 = v44;
  v15 = (_QWORD *)v44[4];
  for ( i = v44 + 3; i != v15; v15 = (_QWORD *)v15[1] )
  {
    if ( !v15 )
    {
      MEMORY[0x20] &= 0xFFFFFFF0;
      BUG();
    }
    v17 = (*(_BYTE *)(v15 - 3) & 0x30) == 0;
    *((_BYTE *)v15 - 24) &= 0xF0u;
    if ( !v17 )
      *((_BYTE *)v15 - 23) |= 0x40u;
  }
  sub_12E54A0(
    v14,
    *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL),
    a3,
    *(_QWORD **)(a1 + 24));
  v18 = *(_QWORD *)(a1 + 24);
  if ( *(_QWORD *)v18 )
  {
    v19 = 0;
    if ( (*(unsigned int (__fastcall **)(_QWORD, _QWORD))v18)(*(_QWORD *)(v18 + 8), 0) )
      goto LABEL_12;
  }
  v25 = (__int64)(v44 + 1);
  v26 = (_QWORD *)v44[2];
  if ( v44 + 1 != v26 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v26 )
          BUG();
        v27 = *(_BYTE *)(v26 - 3) & 0xF;
        if ( v27 != 6 )
        {
          if ( v27 )
          {
            v28 = *(_BYTE *)(v26 - 3) & 0xF0 | 3;
            *((_BYTE *)v26 - 24) = v28;
            if ( (v28 & 0x30) != 0 )
              break;
          }
        }
        v26 = (_QWORD *)v26[1];
        if ( (_QWORD *)v25 == v26 )
          goto LABEL_40;
      }
      *((_BYTE *)v26 - 23) |= 0x40u;
      v26 = (_QWORD *)v26[1];
    }
    while ( (_QWORD *)v25 != v26 );
  }
LABEL_40:
  if ( *(int *)(a3 + 4104) >= 0 )
  {
    v47 = 0;
    v46 = (__int64 *)&v48;
    LOBYTE(v48) = 0;
    v35 = (_DWORD *)sub_1C42D70(4, 4);
    *v35 = 4;
    sub_16D40E0(qword_4FBB3B0, v35);
    if ( !(unsigned __int8)sub_12F5100(*(_QWORD *)(a1 + 32), v44, &v46, 0, *(_QWORD *)(a1 + 24)) )
      **(_BYTE **)(a1 + 40) = 0;
    v36 = *(_QWORD *)(a1 + 24);
    if ( *(_QWORD *)v36 )
    {
      v19 = 0;
      if ( (*(unsigned int (__fastcall **)(_QWORD, _QWORD))v36)(*(_QWORD *)(v36 + 8), 0) )
      {
        if ( v46 != (__int64 *)&v48 )
        {
          v19 = (__int64)(v48 + 1);
          j_j___libc_free_0(v46, v48 + 1);
        }
LABEL_12:
        v20 = v44;
        if ( v44 )
        {
          sub_1633490(v44);
          v19 = 736;
          j_j___libc_free_0(v20, 736);
        }
        result = (unsigned int)v65;
        v22 = v64;
        v23 = &v64[48 * (unsigned int)v65];
        if ( v64 != v23 )
        {
          do
          {
            v23 -= 48;
            v24 = (_BYTE *)*((_QWORD *)v23 + 2);
            result = (__int64)(v23 + 32);
            if ( v24 != v23 + 32 )
            {
              v19 = *((_QWORD *)v23 + 4) + 1LL;
              result = j_j___libc_free_0(v24, v19);
            }
          }
          while ( v22 != v23 );
          v23 = v64;
        }
        if ( v23 != v66 )
          result = _libc_free(v23, v19);
        goto LABEL_21;
      }
    }
    sub_2240CE0(&v46, v47 - 1, 1);
    if ( &_pthread_key_create )
    {
      v38 = pthread_mutex_lock(*(pthread_mutex_t **)(a1 + 48));
      if ( v38 )
        sub_4264C5(v38);
    }
    sub_2241490(*(_QWORD *)(a1 + 56), v46, v47, v37);
    v39 = *(_QWORD *)(a1 + 64);
    v25 = *(_QWORD *)(v39 + 8);
    v45 = v47;
    if ( v25 == *(_QWORD *)(v39 + 16) )
    {
      sub_A235E0(v39, (_BYTE *)v25, &v45);
    }
    else
    {
      if ( v25 )
      {
        *(_QWORD *)v25 = v47;
        v25 = *(_QWORD *)(v39 + 8);
      }
      v25 += 8;
      *(_QWORD *)(v39 + 8) = v25;
    }
    if ( &_pthread_key_create )
      pthread_mutex_unlock(*(pthread_mutex_t **)(a1 + 48));
    if ( v46 != (__int64 *)&v48 )
    {
      v25 = (__int64)(v48 + 1);
      j_j___libc_free_0(v46, v48 + 1);
    }
  }
  v29 = v44;
  v30 = (_QWORD *)(**(_QWORD **)(a1 + 72) + v8);
  v44 = 0;
  *v30 = v29;
  if ( **(_QWORD **)a1 )
  {
    if ( (unsigned int)sub_1682970() )
LABEL_71:
      sub_16BD130("GNU Jobserver support requested, but an error occurred", 1);
  }
  v31 = v44;
  if ( v44 )
  {
    sub_1633490(v44);
    v25 = 736;
    j_j___libc_free_0(v31, 736);
  }
  v32 = v64;
  result = 48LL * (unsigned int)v65;
  v33 = &v64[result];
  if ( v64 != &v64[result] )
  {
    do
    {
      v33 -= 48;
      v34 = (_BYTE *)*((_QWORD *)v33 + 2);
      result = (__int64)(v33 + 32);
      if ( v34 != v33 + 32 )
      {
        v25 = *((_QWORD *)v33 + 4) + 1LL;
        result = j_j___libc_free_0(v34, v25);
      }
    }
    while ( v32 != v33 );
    v33 = v64;
  }
  if ( v33 != v66 )
    result = _libc_free(v33, v25);
LABEL_21:
  if ( v61 )
    result = j_j___libc_free_0(v61, v63 - v61);
  if ( v58 != v60 )
    result = j_j___libc_free_0(v58, v60[0] + 1LL);
  if ( v55 != v57 )
    result = j_j___libc_free_0(v55, v57[0] + 1LL);
  if ( v51 != v53 )
    result = j_j___libc_free_0(v51, v53[0] + 1LL);
  if ( v43 )
    return (*(__int64 (__fastcall **)(_QWORD *))(*v43 + 8LL))(v43);
  return result;
}
