// Function: sub_308F6A0
// Address: 0x308f6a0
//
_QWORD *__fastcall sub_308F6A0(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  int v8; // eax
  __int64 v9; // r8
  __int64 v10; // rax
  int v11; // edx
  bool v12; // al
  int v13; // eax
  __int64 v14; // r14
  int v15; // r13d
  __int64 v16; // rbx
  __int64 v17; // rcx
  __int64 v18; // rax
  int v19; // esi
  int v20; // ebx
  int v21; // eax
  __int64 v22; // rcx
  int v23; // edx
  unsigned int v24; // eax
  int v25; // esi
  _QWORD *result; // rax
  int v27; // eax
  __int64 v28; // rdx
  int v29; // r13d
  int v30; // r14d
  __int64 v31; // rbx
  __int64 v32; // rax
  int v33; // esi
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // r8
  unsigned int v37; // esi
  _DWORD *v38; // rcx
  int v39; // edi
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  _DWORD *v43; // rsi
  int v44; // eax
  bool v45; // al
  int v46; // edx
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // ecx
  int *v50; // rsi
  int v51; // r9d
  unsigned __int16 v52; // ax
  __int64 *v53; // rax
  __int64 v54; // rdi
  __int64 *v55; // rbx
  __int64 v56; // rdx
  char *v57; // r13
  unsigned __int64 v58; // rax
  char *v59; // rbx
  __int64 v60; // rdx
  const char *v61; // rbx
  int *v62; // rcx
  int v63; // eax
  __int64 v64; // rax
  int v65; // edx
  int v66; // ecx
  int v67; // r10d
  int v68; // edi
  int v69; // esi
  int v70; // r11d
  __int64 v71; // rax
  int v72; // [rsp+4h] [rbp-DCh]
  int *v73; // [rsp+8h] [rbp-D8h]
  int v74; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v75; // [rsp+18h] [rbp-C8h]
  _DWORD *v76; // [rsp+18h] [rbp-C8h]
  _DWORD *v77; // [rsp+18h] [rbp-C8h]
  __int64 v79; // [rsp+28h] [rbp-B8h] BYREF
  char *endptr; // [rsp+38h] [rbp-A8h] BYREF
  char **v81; // [rsp+40h] [rbp-A0h] BYREF
  size_t v82; // [rsp+48h] [rbp-98h]
  char *v83[2]; // [rsp+50h] [rbp-90h] BYREF
  char *s[2]; // [rsp+60h] [rbp-80h] BYREF
  char *v85[2]; // [rsp+70h] [rbp-70h] BYREF
  char *nptr[2]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD v87[10]; // [rsp+90h] [rbp-50h] BYREF

  v8 = *(unsigned __int16 *)(a2 + 68);
  v9 = a2;
  v79 = a2;
  if ( (unsigned int)(v8 - 1) <= 1 && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 0x10) != 0 )
    goto LABEL_32;
  v10 = *(_QWORD *)(a2 + 16);
  v11 = *(_DWORD *)(a2 + 44);
  v75 = *(_BYTE *)(v10 + 4);
  if ( (v11 & 4) != 0 || (v11 & 8) == 0 )
  {
    v12 = (*(_QWORD *)(v10 + 24) & 0x100000LL) != 0;
  }
  else
  {
    v12 = sub_2E88A90(a2, 0x100000, 1);
    v9 = v79;
  }
  if ( v12
    || (unsigned int)*(unsigned __int16 *)(v9 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(v9 + 32) + 64LL) & 8) != 0
    || ((v13 = *(_DWORD *)(v9 + 44), (v13 & 4) != 0) || (v13 & 8) == 0
      ? (v14 = (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) >> 19) & 1LL)
      : (v45 = sub_2E88A90(v9, 0x80000, 1), v9 = v79, LOBYTE(v14) = v45),
        (_BYTE)v14) )
  {
LABEL_32:
    v27 = sub_3089AC0(a1, v9);
    v28 = v79;
    v29 = v27;
    v30 = *(_DWORD *)(v79 + 40) & 0xFFFFFF;
    if ( !v30 )
      goto LABEL_29;
    v31 = 0;
    while ( 1 )
    {
      v32 = *(_QWORD *)(v28 + 32) + 40 * v31;
      if ( *(_BYTE *)v32 )
      {
        if ( *(_BYTE *)v32 == 5 && (!v29 || v29 + 6 != (_DWORD)v31) )
        {
          v33 = *(_DWORD *)(v32 + 24);
LABEL_42:
          sub_308E6E0(a1, v33);
        }
      }
      else if ( (*(_BYTE *)(v32 + 3) & 0x10) == 0 )
      {
        v34 = *(_DWORD *)(v32 + 8);
        v35 = *(unsigned int *)(a3 + 24);
        v36 = *(_QWORD *)(a3 + 8);
        if ( (_DWORD)v35 )
        {
          v37 = (v35 - 1) & (37 * v34);
          v38 = (_DWORD *)(v36 + 8LL * v37);
          v39 = *v38;
          if ( v34 == *v38 )
          {
LABEL_45:
            if ( v38 != (_DWORD *)(v36 + 8 * v35) )
            {
              if ( v29 && v29 + 6 == (_DWORD)v31
                || (v76 = v38,
                    sub_308E6E0(a1, v38[1]),
                    v38 = v76,
                    v76 != (_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL * *(unsigned int *)(a3 + 24))) )
              {
                v33 = v38[1];
                if ( v33 >= (int)(-858993459
                                * ((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 16LL)
                                           - *(_QWORD *)(*(_QWORD *)(a1 + 440) + 8LL)) >> 3)) )
                {
                  if ( (unsigned int)*(unsigned __int16 *)(v79 + 68) - 1 <= 1
                    && (*(_BYTE *)(*(_QWORD *)(v79 + 32) + 64LL) & 0x10) != 0 )
                  {
                    goto LABEL_42;
                  }
                  v40 = *(_DWORD *)(v79 + 44);
                  if ( (v40 & 4) == 0 && (v40 & 8) != 0 )
                  {
                    v77 = v38;
                    LOBYTE(v41) = sub_2E88A90(v79, 0x100000, 1);
                    v38 = v77;
                  }
                  else
                  {
                    v41 = (*(_QWORD *)(*(_QWORD *)(v79 + 16) + 24LL) >> 20) & 1LL;
                  }
                  if ( (_BYTE)v41
                    || (v42 = *(_QWORD *)(v79 + 48),
                        v43 = (_DWORD *)(v42 & 0xFFFFFFFFFFFFFFF8LL),
                        (v42 & 0xFFFFFFFFFFFFFFF8LL) == 0) )
                  {
LABEL_95:
                    v33 = v38[1];
                    goto LABEL_42;
                  }
                  v44 = v42 & 7;
                  if ( v44 )
                  {
                    if ( v44 != 3 || *v43 != 1 )
                      goto LABEL_95;
                  }
                  else
                  {
                    *(_QWORD *)(v79 + 48) = v43;
                  }
                }
              }
            }
          }
          else
          {
            v66 = 1;
            while ( v39 != -1 )
            {
              v67 = v66 + 1;
              v37 = (v35 - 1) & (v66 + v37);
              v38 = (_DWORD *)(v36 + 8LL * v37);
              v39 = *v38;
              if ( v34 == *v38 )
                goto LABEL_45;
              v66 = v67;
            }
          }
        }
      }
      if ( v30 == (_DWORD)++v31 )
        goto LABEL_29;
      v28 = v79;
    }
  }
  if ( a4 || (unsigned int)*(unsigned __int16 *)(v9 + 68) - 2907 > 1 )
  {
    v15 = *(_DWORD *)(v9 + 40) & 0xFFFFFF;
    if ( !v15 )
      goto LABEL_29;
    v16 = 0;
    while ( 1 )
    {
      v17 = *(_QWORD *)(v9 + 32);
      v18 = v17 + 40 * v16;
      if ( *(_BYTE *)v18 )
      {
        if ( *(_BYTE *)v18 != 5 )
          goto LABEL_16;
        v19 = *(_DWORD *)(v18 + 24);
        LODWORD(endptr) = v19;
        if ( (unsigned int)*(unsigned __int16 *)(v9 + 68) - 2679 <= 1 )
        {
          if ( v16 == 1 )
          {
            if ( *(_BYTE *)(v17 + 80) == 1 && !*(_QWORD *)(v17 + 104) )
            {
LABEL_69:
              LOBYTE(v14) = 1;
              goto LABEL_16;
            }
          }
          else if ( (_DWORD)v16 == 2 && *(_BYTE *)(v17 + 40) == 1 && !*(_QWORD *)(v17 + 64) )
          {
            goto LABEL_69;
          }
        }
      }
      else
      {
        if ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 )
          goto LABEL_16;
        v46 = *(_DWORD *)(v18 + 8);
        v47 = *(unsigned int *)(a3 + 24);
        v48 = *(_QWORD *)(a3 + 8);
        if ( !(_DWORD)v47 )
          goto LABEL_16;
        v49 = (v47 - 1) & (37 * v46);
        v50 = (int *)(v48 + 8LL * v49);
        v51 = *v50;
        if ( *v50 != v46 )
        {
          v69 = 1;
          while ( v51 != -1 )
          {
            v70 = v69 + 1;
            v49 = (v47 - 1) & (v69 + v49);
            v50 = (int *)(v48 + 8LL * v49);
            v51 = *v50;
            if ( v46 == *v50 )
              goto LABEL_66;
            v69 = v70;
          }
LABEL_16:
          if ( v15 == (_DWORD)++v16 )
            goto LABEL_24;
          goto LABEL_17;
        }
LABEL_66:
        if ( v50 == (int *)(v48 + 8 * v47) )
          goto LABEL_16;
        v19 = v50[1];
        LODWORD(endptr) = v19;
        v52 = *(_WORD *)(v9 + 68);
        if ( v52 <= 0x1B57u )
        {
          if ( v52 > 0x1B55u )
            goto LABEL_69;
        }
        else if ( (unsigned __int16)(v52 - 7006) <= 1u )
        {
          goto LABEL_69;
        }
      }
      ++v16;
      sub_308E6E0(a1, v19);
      if ( v15 == (_DWORD)v16 )
      {
LABEL_24:
        if ( (_BYTE)v14 )
          goto LABEL_25;
        goto LABEL_29;
      }
LABEL_17:
      v9 = v79;
    }
  }
  v56 = -1;
  v57 = *(char **)(*(_QWORD *)(v9 + 32) + 64LL);
  v81 = v83;
  if ( v57 )
    v56 = (__int64)&v57[strlen(v57)];
  sub_3089B20((__int64 *)&v81, v57, v56);
  v58 = sub_2241950((__int64 *)&v81, "_param_", 0xFFFFFFFFFFFFFFFFLL, 7u) + 1;
  if ( v58 > v82 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::substr", v58, v82);
  s[0] = (char *)v85;
  sub_3089B20((__int64 *)s, (_BYTE *)v81 + v58, (__int64)v81 + v82);
  v59 = s[0];
  nptr[0] = (char *)v87;
  v60 = -1;
  if ( s[0] )
    v60 = (__int64)&v59[strlen(s[0])];
  sub_3089B20((__int64 *)nptr, v59, v60);
  v61 = nptr[0];
  v62 = __errno_location();
  v63 = *v62;
  *v62 = 0;
  v73 = v62;
  v72 = v63;
  v64 = strtol(v61, &endptr, 10);
  v65 = v64;
  if ( v61 == endptr )
    sub_426290((__int64)"stoi");
  if ( *v73 == 34 || (unsigned __int64)(v64 + 0x80000000LL) > 0xFFFFFFFF )
    sub_426320((__int64)"stoi");
  if ( !*v73 )
    *v73 = v72;
  if ( (_QWORD *)nptr[0] != v87 )
  {
    v74 = v64;
    j_j___libc_free_0((unsigned __int64)nptr[0]);
    v65 = v74;
  }
  LODWORD(endptr) = v65
                  - 858993459
                  * ((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 16LL) - *(_QWORD *)(*(_QWORD *)(a1 + 440) + 8LL)) >> 3);
  if ( (char **)s[0] != v85 )
    j_j___libc_free_0((unsigned __int64)s[0]);
  if ( v81 != v83 )
    j_j___libc_free_0((unsigned __int64)v81);
LABEL_25:
  v20 = (int)endptr;
  if ( (_DWORD)endptr == -1 )
  {
LABEL_29:
    result = (_QWORD *)a5;
    if ( a5 )
      return (_QWORD *)sub_308F390(a1, v79, a3, a5);
  }
  else
  {
    if ( *(_DWORD *)(a1 + 312) )
    {
      v21 = *(_DWORD *)(a1 + 320);
      v22 = *(_QWORD *)(a1 + 304);
      if ( v21 )
      {
        v23 = v21 - 1;
        v24 = (v21 - 1) & (37 * (_DWORD)endptr);
        v25 = *(_DWORD *)(v22 + 4LL * v24);
        if ( (_DWORD)endptr == v25 )
          goto LABEL_29;
        v68 = 1;
        while ( v25 != -1 )
        {
          v24 = v23 & (v68 + v24);
          v25 = *(_DWORD *)(v22 + 4LL * v24);
          if ( (_DWORD)endptr == v25 )
            goto LABEL_29;
          ++v68;
        }
      }
    }
    if ( v75 > 1u )
      goto LABEL_29;
    LODWORD(s[0]) = *(_DWORD *)(*(_QWORD *)(v79 + 32) + 8LL);
    *sub_307C5F0(a3, (int *)s) = v20;
    if ( a5 )
      sub_22B6470((__int64)nptr, a5, (int *)s);
    v53 = (__int64 *)sub_308C8B0(a1 + 328, (int *)&endptr);
    v54 = *v53;
    v55 = v53;
    if ( !*v53 )
    {
      v71 = sub_22077B0(0x20u);
      v54 = v71;
      if ( v71 )
      {
        *(_QWORD *)v71 = 0;
        *(_QWORD *)(v71 + 8) = 0;
        *(_QWORD *)(v71 + 16) = 0;
        *(_DWORD *)(v71 + 24) = 0;
      }
      *v55 = v71;
    }
    result = sub_2E263C0(v54, &v79);
    *(_DWORD *)result = 1;
  }
  return result;
}
