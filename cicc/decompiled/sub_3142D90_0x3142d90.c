// Function: sub_3142D90
// Address: 0x3142d90
//
_QWORD *__fastcall sub_3142D90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        _QWORD *a7,
        size_t a8)
{
  unsigned __int8 v12; // al
  __int64 *v13; // rdx
  __int64 v14; // r15
  __int64 i; // rax
  unsigned __int8 v16; // r8
  __int64 v17; // rdx
  __int64 *v18; // rdx
  __int64 v19; // rax
  _QWORD *result; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  int v23; // esi
  int v24; // edx
  unsigned int v25; // esi
  unsigned int v26; // r9d
  __int64 v27; // r8
  __int64 v28; // r13
  int v29; // eax
  int v30; // r11d
  __int64 v31; // r10
  __int64 v32; // rcx
  int v33; // r9d
  unsigned int j; // r13d
  const void *v35; // rsi
  unsigned int v36; // r13d
  __int64 v37; // r13
  int v38; // eax
  int v39; // r11d
  int v40; // r9d
  unsigned int k; // r10d
  __int64 v42; // rcx
  const void *v43; // rsi
  int v44; // eax
  bool v45; // al
  __int64 v46; // rbx
  __int64 v47; // r12
  unsigned int v48; // esi
  int v49; // eax
  int v50; // eax
  __int64 v51; // rdx
  unsigned int v52; // r10d
  int v53; // eax
  int v54; // eax
  __int64 v55; // r13
  int v56; // eax
  int v57; // r11d
  __int64 v58; // rcx
  int v59; // r9d
  unsigned int v60; // r13d
  const void *v61; // rsi
  unsigned int v62; // r13d
  int v63; // eax
  bool v64; // al
  int v65; // eax
  bool v66; // al
  __int64 v67; // [rsp+0h] [rbp-A0h]
  int v68; // [rsp+0h] [rbp-A0h]
  int v69; // [rsp+0h] [rbp-A0h]
  __int64 v70; // [rsp+8h] [rbp-98h]
  __int64 v71; // [rsp+8h] [rbp-98h]
  __int64 v72; // [rsp+10h] [rbp-90h]
  int v73; // [rsp+10h] [rbp-90h]
  int v74; // [rsp+10h] [rbp-90h]
  int v75; // [rsp+18h] [rbp-88h]
  __int64 v76; // [rsp+18h] [rbp-88h]
  __int64 v77; // [rsp+18h] [rbp-88h]
  unsigned int v78; // [rsp+20h] [rbp-80h]
  __int64 v79; // [rsp+20h] [rbp-80h]
  __int64 v80; // [rsp+20h] [rbp-80h]
  int v82; // [rsp+38h] [rbp-68h]
  int v83; // [rsp+38h] [rbp-68h]
  int v84; // [rsp+38h] [rbp-68h]
  int v85; // [rsp+38h] [rbp-68h]
  _QWORD *v86; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v87; // [rsp+48h] [rbp-58h] BYREF
  __int64 v88; // [rsp+50h] [rbp-50h] BYREF
  __int64 v89; // [rsp+58h] [rbp-48h]
  __int64 v90; // [rsp+60h] [rbp-40h]

  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(__int64 **)(a2 - 32);
  else
    v13 = (__int64 *)(a2 - 16 - 8LL * ((v12 >> 2) & 0xF));
  v14 = *v13;
  for ( i = sub_B10CD0(a3); ; i = v22 )
  {
    v16 = *(_BYTE *)(i - 16);
    if ( (v16 & 2) != 0 )
    {
      v17 = *(_QWORD *)(i - 32);
      if ( *(_DWORD *)(i - 24) != 2 )
        goto LABEL_6;
    }
    else
    {
      v21 = i - 16;
      if ( ((*(_WORD *)(i - 16) >> 6) & 0xF) != 2 )
        goto LABEL_16;
      v17 = v21 - 8LL * ((v16 >> 2) & 0xF);
    }
    v22 = *(_QWORD *)(v17 + 8);
    if ( !v22 )
      break;
  }
  v21 = i - 16;
  if ( (*(_BYTE *)(i - 16) & 2) != 0 )
  {
LABEL_6:
    v18 = *(__int64 **)(i - 32);
    goto LABEL_7;
  }
LABEL_16:
  v18 = (__int64 *)(v21 - 8LL * ((v16 >> 2) & 0xF));
LABEL_7:
  v19 = *v18;
  v88 = a2;
  v90 = v14;
  v89 = v19;
  result = (_QWORD *)sub_3140F10(a4, &v88, &v86);
  if ( !(_BYTE)result )
  {
    v23 = *(_DWORD *)(a4 + 16);
    result = v86;
    ++*(_QWORD *)a4;
    v24 = v23 + 1;
    v25 = *(_DWORD *)(a4 + 24);
    v87 = result;
    if ( 4 * v24 >= 3 * v25 )
    {
      v25 *= 2;
    }
    else if ( v25 - *(_DWORD *)(a4 + 20) - v24 > v25 >> 3 )
    {
LABEL_19:
      *(_DWORD *)(a4 + 16) = v24;
      if ( result[2] != -4096 || result[1] != -4096 || *result != -4096 )
        --*(_DWORD *)(a4 + 20);
      result[2] = v90;
      result[1] = v89;
      *result = v88;
      if ( !a6 )
        return result;
      goto LABEL_22;
    }
    sub_3142BA0(a4, v25);
    sub_3140F10(a4, &v88, &v87);
    v24 = *(_DWORD *)(a4 + 16) + 1;
    result = v87;
    goto LABEL_19;
  }
  if ( !a6 )
    return result;
LABEL_22:
  v26 = *(_DWORD *)(a5 + 24);
  if ( !v26 )
  {
    ++*(_QWORD *)a5;
LABEL_24:
    sub_3141930(a5, 2 * v26);
    v27 = 0;
    v82 = *(_DWORD *)(a5 + 24);
    if ( !v82 )
      goto LABEL_60;
    v28 = *(_QWORD *)(a5 + 8);
    v29 = sub_C94890(a7, a8);
    v30 = 1;
    v31 = 0;
    v32 = v28;
    v33 = v82 - 1;
    for ( j = (v82 - 1) & v29; ; j = v33 & v36 )
    {
      v27 = v32 + 48LL * j;
      v35 = *(const void **)v27;
      if ( *(_QWORD *)v27 == -1 )
        goto LABEL_57;
      if ( v35 == (const void *)-2LL )
      {
        v64 = (_QWORD *)((char *)a7 + 2) == 0;
      }
      else
      {
        if ( *(_QWORD *)(v27 + 8) != a8 )
          goto LABEL_29;
        v68 = v30;
        v70 = v31;
        v73 = v33;
        v76 = v32;
        if ( !a8 )
          goto LABEL_60;
        v79 = v32 + 48LL * j;
        v63 = memcmp(a7, v35, a8);
        v27 = v79;
        v32 = v76;
        v33 = v73;
        v31 = v70;
        v30 = v68;
        v64 = v63 == 0;
      }
      if ( v64 )
        goto LABEL_60;
      if ( !v31 && v35 == (const void *)-2LL )
        v31 = v27;
LABEL_29:
      v36 = v30 + j;
      ++v30;
    }
  }
  v83 = *(_DWORD *)(a5 + 24);
  v37 = *(_QWORD *)(a5 + 8);
  v38 = sub_C94890(a7, a8);
  v39 = 1;
  v27 = 0;
  v40 = v83 - 1;
  for ( k = (v83 - 1) & v38; ; k = v40 & v52 )
  {
    v42 = v37 + 48LL * k;
    v43 = *(const void **)v42;
    if ( *(_QWORD *)v42 == -1 )
    {
      v45 = (_QWORD *)((char *)a7 + 1) == 0;
    }
    else if ( v43 == (const void *)-2LL )
    {
      v45 = (_QWORD *)((char *)a7 + 2) == 0;
    }
    else
    {
      if ( *(_QWORD *)(v42 + 8) != a8 )
        goto LABEL_49;
      v72 = v27;
      v75 = v39;
      v78 = k;
      v84 = v40;
      if ( !a8 )
        goto LABEL_37;
      v67 = v37 + 48LL * k;
      v44 = memcmp(a7, v43, a8);
      v42 = v67;
      v40 = v84;
      k = v78;
      v39 = v75;
      v27 = v72;
      v45 = v44 == 0;
    }
    if ( v45 )
    {
LABEL_37:
      v46 = v42 + 16;
      goto LABEL_38;
    }
    if ( v43 == (const void *)-1LL )
      break;
LABEL_49:
    if ( v43 != (const void *)-2LL || v27 )
      v42 = v27;
    v52 = v39 + k;
    v27 = v42;
    ++v39;
  }
  v54 = *(_DWORD *)(a5 + 16);
  v26 = *(_DWORD *)(a5 + 24);
  if ( !v27 )
    v27 = v42;
  ++*(_QWORD *)a5;
  v53 = v54 + 1;
  if ( 4 * v53 >= 3 * v26 )
    goto LABEL_24;
  if ( v26 - (v53 + *(_DWORD *)(a5 + 20)) > v26 >> 3 )
    goto LABEL_61;
  sub_3141930(a5, v26);
  v27 = 0;
  v85 = *(_DWORD *)(a5 + 24);
  if ( !v85 )
    goto LABEL_60;
  v55 = *(_QWORD *)(a5 + 8);
  v56 = sub_C94890(a7, a8);
  v57 = 1;
  v31 = 0;
  v58 = v55;
  v59 = v85 - 1;
  v60 = (v85 - 1) & v56;
  while ( 2 )
  {
    v27 = v58 + 48LL * v60;
    v61 = *(const void **)v27;
    if ( *(_QWORD *)v27 != -1 )
    {
      if ( v61 == (const void *)-2LL )
      {
        v66 = (_QWORD *)((char *)a7 + 2) == 0;
      }
      else
      {
        if ( *(_QWORD *)(v27 + 8) != a8 )
        {
LABEL_76:
          if ( v31 || v61 != (const void *)-2LL )
            v27 = v31;
          v62 = v57 + v60;
          v31 = v27;
          ++v57;
          v60 = v59 & v62;
          continue;
        }
        v69 = v57;
        v71 = v31;
        v74 = v59;
        v77 = v58;
        if ( !a8 )
          goto LABEL_60;
        v80 = v58 + 48LL * v60;
        v65 = memcmp(a7, v61, a8);
        v27 = v80;
        v58 = v77;
        v59 = v74;
        v31 = v71;
        v57 = v69;
        v66 = v65 == 0;
      }
      if ( v66 )
        goto LABEL_60;
      if ( v61 == (const void *)-1LL )
        goto LABEL_58;
      goto LABEL_76;
    }
    break;
  }
LABEL_57:
  if ( a7 == (_QWORD *)-1LL )
    goto LABEL_60;
LABEL_58:
  if ( v31 )
    v27 = v31;
LABEL_60:
  v53 = *(_DWORD *)(a5 + 16) + 1;
LABEL_61:
  *(_DWORD *)(a5 + 16) = v53;
  if ( *(_QWORD *)v27 != -1 )
    --*(_DWORD *)(a5 + 20);
  *(_QWORD *)(v27 + 16) = 0;
  v46 = v27 + 16;
  *(_QWORD *)(v27 + 24) = 0;
  *(_QWORD *)v27 = a7;
  *(_QWORD *)(v27 + 32) = 0;
  *(_QWORD *)(v27 + 8) = a8;
  *(_DWORD *)(v27 + 40) = 0;
LABEL_38:
  v47 = sub_B10D40(a3);
  result = (_QWORD *)sub_3140DC0(v46, &v88, &v86);
  if ( !(_BYTE)result )
  {
    v87 = v86;
    v48 = *(_DWORD *)(v46 + 24);
    v49 = *(_DWORD *)(v46 + 16);
    ++*(_QWORD *)v46;
    v50 = v49 + 1;
    if ( 4 * v50 >= 3 * v48 )
    {
      v48 *= 2;
    }
    else if ( v48 - *(_DWORD *)(v46 + 20) - v50 > v48 >> 3 )
    {
LABEL_41:
      *(_DWORD *)(v46 + 16) = v50;
      result = v87;
      if ( v87[2] != -4096 || v87[1] != -4096 || *v87 != -4096 )
        --*(_DWORD *)(v46 + 20);
      v51 = v90;
      result[3] = v47;
      result[2] = v51;
      result[1] = v89;
      *result = v88;
      return result;
    }
    sub_3141720(v46, v48);
    sub_3140DC0(v46, &v88, &v87);
    v50 = *(_DWORD *)(v46 + 16) + 1;
    goto LABEL_41;
  }
  return result;
}
