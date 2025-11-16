// Function: sub_2427400
// Address: 0x2427400
//
__int64 *__fastcall sub_2427400(__int64 *a1, __int64 a2, _BYTE *a3, int a4)
{
  __int64 v6; // rax
  __int64 v7; // r14
  int v8; // r13d
  unsigned int v9; // r15d
  __int64 v10; // rax
  unsigned __int8 v11; // si
  int v12; // edx
  unsigned __int8 v13; // al
  _BYTE **v14; // rbx
  unsigned __int64 v15; // rbx
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // r10
  _BYTE *v19; // rdi
  __int64 *v20; // r9
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  const void *v25; // r14
  size_t v26; // rbx
  unsigned __int64 v27; // rax
  const char *v28; // rax
  char *v29; // rsi
  unsigned __int64 v30; // rdx
  const char *v31; // rdi
  unsigned __int8 v33; // al
  __int64 *v34; // rbx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // r9
  const void *v38; // r8
  size_t v39; // r13
  char *v40; // rdi
  const char *v41; // rax
  __int64 v42; // rax
  _BYTE *v43; // r15
  __int64 v44; // rdx
  __int64 v45; // r9
  _BYTE *v46; // r8
  _BYTE *v47; // r9
  __int64 v48; // rdx
  _BYTE *v49; // rsi
  char *v50; // rdi
  char *v51; // rsi
  unsigned __int64 v52; // rdx
  void *src; // [rsp+0h] [rbp-240h]
  __int64 v55; // [rsp+8h] [rbp-238h]
  _QWORD v56[4]; // [rsp+10h] [rbp-230h] BYREF
  __int16 v57; // [rsp+30h] [rbp-210h]
  char v58[32]; // [rsp+40h] [rbp-200h] BYREF
  __int16 v59; // [rsp+60h] [rbp-1E0h]
  char v60[32]; // [rsp+70h] [rbp-1D0h] BYREF
  __int16 v61; // [rsp+90h] [rbp-1B0h]
  char v62[32]; // [rsp+A0h] [rbp-1A0h] BYREF
  __int16 v63; // [rsp+C0h] [rbp-180h]
  const char *v64; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v65; // [rsp+D8h] [rbp-168h]
  __int64 v66; // [rsp+E0h] [rbp-160h]
  _BYTE v67[136]; // [rsp+E8h] [rbp-158h] BYREF
  char *v68; // [rsp+170h] [rbp-D0h] BYREF
  unsigned __int64 v69; // [rsp+178h] [rbp-C8h]
  __int64 v70; // [rsp+180h] [rbp-C0h]
  _BYTE dest[184]; // [rsp+188h] [rbp-B8h] BYREF

  v6 = sub_BA8DC0(a2, (__int64)"llvm.gcov", 9);
  if ( !v6 || (v7 = v6, (v8 = sub_B91A00(v6)) == 0) )
  {
LABEL_8:
    if ( *a3 != 16
      && ((v13 = *(a3 - 16), (v13 & 2) != 0)
        ? (v14 = (_BYTE **)*((_QWORD *)a3 - 4))
        : (v14 = (_BYTE **)&a3[-8 * ((v13 >> 2) & 0xF) - 16]),
          (a3 = *v14) == 0)
      || ((v33 = *(a3 - 16), (v33 & 2) != 0)
        ? (v34 = (__int64 *)*((_QWORD *)a3 - 4))
        : (v34 = (__int64 *)&a3[-8 * ((v33 >> 2) & 0xF) - 16]),
          !*v34) )
    {
      v15 = 0;
      v65 = 0;
      v64 = v67;
      v66 = 128;
      goto LABEL_40;
    }
    v35 = sub_B91420(*v34);
    v65 = 0;
    v64 = v67;
    v38 = (const void *)v35;
    v39 = v36;
    v15 = v36;
    v66 = 128;
    if ( v36 <= 0x80 )
    {
      if ( !v36 )
      {
LABEL_40:
        v65 += v15;
        v41 = "gcno";
        dest[9] = 1;
        dest[8] = 3;
        if ( a4 )
          v41 = "gcda";
        v68 = (char *)v41;
        sub_C80880((__int64)&v64, (const char **)&v68, 0);
        v42 = sub_C80C60((__int64)v64, v65, 0);
        v68 = dest;
        v43 = (_BYTE *)v42;
        v55 = v44;
        v69 = 0;
        v70 = 128;
        if ( (unsigned int)sub_C82800(&v68) )
        {
          *a1 = (__int64)(a1 + 2);
          sub_2425560(a1, v43, (__int64)&v43[v55]);
        }
        else
        {
          v63 = 257;
          v61 = 257;
          v59 = 257;
          v57 = 261;
          v56[1] = v55;
          v56[0] = v43;
          sub_C81B70(&v68, (__int64)v56, (__int64)v58, (__int64)v60, (__int64)v62);
          v51 = v68;
          v52 = v69;
          *a1 = (__int64)(a1 + 2);
          sub_2425560(a1, v51, (__int64)&v51[v52]);
        }
        if ( v68 != dest )
          _libc_free((unsigned __int64)v68);
        v31 = v64;
        if ( v64 != v67 )
          goto LABEL_29;
        return a1;
      }
      v40 = v67;
    }
    else
    {
      src = (void *)v35;
      sub_C8D290((__int64)&v64, v67, v36, 1u, v35, v37);
      v38 = src;
      v40 = (char *)&v64[v65];
    }
    memcpy(v40, v38, v39);
    goto LABEL_40;
  }
  v9 = 0;
  while ( 1 )
  {
    v10 = sub_B91A10(v7, v9);
    v11 = *(_BYTE *)(v10 - 16);
    if ( (v11 & 2) != 0 )
    {
      v12 = *(_DWORD *)(v10 - 24);
      if ( v12 == 3 )
      {
        v16 = v10 - 16;
        v17 = 16;
      }
      else
      {
        if ( v12 != 2 )
          goto LABEL_7;
        v16 = v10 - 16;
        v17 = 8;
      }
      v18 = *(_QWORD *)(v10 - 32);
    }
    else
    {
      v12 = (*(_WORD *)(v10 - 16) >> 6) & 0xF;
      if ( v12 == 3 )
      {
        v16 = v10 - 16;
        v17 = 16;
      }
      else
      {
        if ( v12 != 2 )
          goto LABEL_7;
        v16 = v10 - 16;
        v17 = 8;
      }
      v18 = v16 - 8LL * ((v11 >> 2) & 0xF);
    }
    v19 = *(_BYTE **)(v18 + v17);
    if ( (unsigned __int8)(*v19 - 5) >= 0x20u )
      v19 = 0;
    if ( a3 != v19 )
      goto LABEL_7;
    if ( v12 != 3 )
      break;
    if ( (*(_BYTE *)(v10 - 16) & 2) != 0 )
    {
      v45 = *(_QWORD *)(v10 - 32);
      v46 = *(_BYTE **)v45;
      if ( **(_BYTE **)v45 )
        goto LABEL_7;
    }
    else
    {
      v45 = v16 - 8LL * ((v11 >> 2) & 0xF);
      v46 = *(_BYTE **)v45;
      if ( **(_BYTE **)v45 )
        goto LABEL_7;
    }
    v47 = *(_BYTE **)(v45 + 8);
    if ( !*v47 )
    {
      if ( a4 )
        v49 = (_BYTE *)sub_B91420((__int64)v47);
      else
        v49 = (_BYTE *)sub_B91420((__int64)v46);
      *a1 = (__int64)(a1 + 2);
      sub_2425560(a1, v49, (__int64)&v49[v48]);
      return a1;
    }
LABEL_7:
    if ( v8 == ++v9 )
      goto LABEL_8;
  }
  if ( (*(_BYTE *)(v10 - 16) & 2) != 0 )
    v20 = *(__int64 **)(v10 - 32);
  else
    v20 = (__int64 *)(v16 - 8LL * ((v11 >> 2) & 0xF));
  if ( *(_BYTE *)*v20 )
    goto LABEL_7;
  v21 = sub_B91420(*v20);
  v68 = dest;
  v25 = (const void *)v21;
  v26 = v22;
  v27 = v22;
  v69 = 0;
  v70 = 128;
  if ( v22 > 0x80 )
  {
    sub_C8D290((__int64)&v68, dest, v22, 1u, v23, v24);
    v50 = &v68[v69];
  }
  else
  {
    if ( !v22 )
      goto LABEL_26;
    v50 = dest;
  }
  memcpy(v50, v25, v26);
  v27 = v26 + v69;
LABEL_26:
  v69 = v27;
  v28 = "gcno";
  v67[9] = 1;
  v67[8] = 3;
  if ( a4 )
    v28 = "gcda";
  v64 = v28;
  sub_C80880((__int64)&v68, &v64, 0);
  v29 = v68;
  v30 = v69;
  *a1 = (__int64)(a1 + 2);
  sub_2425560(a1, v29, (__int64)&v29[v30]);
  v31 = v68;
  if ( v68 != dest )
LABEL_29:
    _libc_free((unsigned __int64)v31);
  return a1;
}
