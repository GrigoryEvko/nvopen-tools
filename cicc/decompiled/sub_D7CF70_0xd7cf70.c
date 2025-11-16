// Function: sub_D7CF70
// Address: 0xd7cf70
//
__int64 __fastcall sub_D7CF70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 result; // rax
  _BYTE *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r11
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // edx
  int v22; // edx
  __int64 v23; // r8
  int v24; // r13d
  _QWORD *v25; // r9
  unsigned int v26; // eax
  _QWORD *v27; // rcx
  __int64 v28; // rdi
  unsigned __int8 *v29; // rsi
  __int64 *v30; // r14
  __int64 *v31; // r13
  __int64 v32; // rdi
  __int64 v33; // rdi
  _BYTE *v34; // rdi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 *v38; // r12
  __int64 *v39; // rbx
  __int64 v40; // rdi
  __int64 v41; // rcx
  int v42; // edx
  int v43; // edx
  __int64 v44; // r8
  int v45; // r11d
  unsigned __int8 *v46; // r9
  unsigned int v47; // eax
  __int64 v48; // rdi
  _QWORD *v49; // r8
  __int64 v50; // rax
  __int64 v51; // rdi
  _QWORD *v52; // rax
  _QWORD *v53; // r8
  __int64 v54; // rax
  _BYTE *v55; // rsi
  __int64 v59; // [rsp+10h] [rbp-100h]
  __int64 v61; // [rsp+18h] [rbp-F8h]
  char v62; // [rsp+27h] [rbp-E9h] BYREF
  __int64 v63; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v64[2]; // [rsp+30h] [rbp-E0h] BYREF
  _BYTE v65[32]; // [rsp+40h] [rbp-D0h] BYREF
  _BYTE *v66; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v67; // [rsp+68h] [rbp-A8h]
  _BYTE v68[32]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v69; // [rsp+90h] [rbp-80h] BYREF
  __int64 v70; // [rsp+98h] [rbp-78h]
  _BYTE v71[112]; // [rsp+A0h] [rbp-70h] BYREF

  v7 = *(_QWORD *)(a1 - 32);
  if ( !v7 || *(_BYTE *)v7 || *(_QWORD *)(a1 + 80) != *(_QWORD *)(v7 + 24) )
    BUG();
  result = *(unsigned int *)(v7 + 36);
  if ( (unsigned int)result > 0x165 )
  {
    if ( (_DWORD)result != 358 )
      return result;
LABEL_7:
    result = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
    v12 = *(_BYTE **)(result + 24);
    if ( *v12 )
      return result;
    v13 = sub_B91420((__int64)v12);
    v15 = sub_B2F650(v13, v14);
    v16 = a1;
    v64[0] = v15;
    v18 = v15;
    v19 = *(_QWORD *)(a1 + 16);
    if ( v19 )
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(v19 + 24);
        if ( *(_BYTE *)v20 != 85 )
          break;
        v41 = *(_QWORD *)(v20 - 32);
        if ( !v41
          || *(_BYTE *)v41
          || *(_QWORD *)(v41 + 24) != *(_QWORD *)(v20 + 80)
          || (*(_BYTE *)(v41 + 33) & 0x20) == 0
          || *(_DWORD *)(v41 + 36) != 11 )
        {
          break;
        }
        v19 = *(_QWORD *)(v19 + 8);
        if ( !v19 )
          goto LABEL_12;
      }
      v21 = *(_DWORD *)(a2 + 24);
      if ( v21 )
      {
        v22 = v21 - 1;
        v23 = *(_QWORD *)(a2 + 8);
        v24 = 1;
        v25 = 0;
        v26 = v22 & (((0xBF58476D1CE4E5B9LL * v18) >> 31) ^ (484763065 * v18));
        v27 = (_QWORD *)(v23 + 8LL * v26);
        v28 = *v27;
        if ( v18 == *v27 )
          goto LABEL_12;
        while ( v28 != -1 )
        {
          if ( !v25 && v28 == -2 )
            v25 = v27;
          v26 = v22 & (v24 + v26);
          v27 = (_QWORD *)(v23 + 8LL * v26);
          v28 = *v27;
          if ( v18 == *v27 )
            goto LABEL_12;
          ++v24;
        }
        if ( !v25 )
          v25 = v27;
      }
      else
      {
        v25 = 0;
      }
      v51 = a2;
      v59 = a1;
      v61 = a2;
      v52 = sub_D7AAE0(v51, v64, v25);
      v16 = v59;
      v53 = v52;
      v54 = v64[0];
      *v53 = v64[0];
      v55 = *(_BYTE **)(v61 + 40);
      if ( v55 == *(_BYTE **)(v61 + 48) )
      {
        sub_9CA200(v61 + 32, v55, v64);
        v16 = v59;
      }
      else
      {
        if ( v55 )
        {
          *(_QWORD *)v55 = v54;
          v55 = *(_BYTE **)(v61 + 40);
        }
        *(_QWORD *)(v61 + 40) = v55 + 8;
      }
    }
LABEL_12:
    v29 = (unsigned __int8 *)&v66;
    v69 = (__int64 *)v71;
    v70 = 0x400000000LL;
    v66 = v68;
    v67 = 0x400000000LL;
    result = sub_E02490(&v69, &v66, v16, a7);
    v30 = v69;
    v31 = &v69[2 * (unsigned int)v70];
    if ( v31 != v69 )
    {
      do
      {
        v32 = *v30;
        v29 = (unsigned __int8 *)v30[1];
        v30 += 2;
        result = sub_D7CB10(v32, v29, v64[0], a3, a5);
      }
      while ( v31 != v30 );
    }
    v33 = (__int64)v66;
    if ( v66 == v68 )
      goto LABEL_16;
    goto LABEL_15;
  }
  if ( (unsigned int)result <= 0x163 )
  {
    if ( (_DWORD)result != 300 )
      return result;
    goto LABEL_7;
  }
  result = *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v34 = *(_BYTE **)(result + 24);
  if ( *v34 )
    return result;
  v36 = sub_B91420((__int64)v34);
  v63 = sub_B2F650(v36, v37);
  v29 = (unsigned __int8 *)v64;
  v69 = (__int64 *)v71;
  v70 = 0x400000000LL;
  v64[0] = (__int64)v65;
  v64[1] = 0x400000000LL;
  v66 = v68;
  v67 = 0x400000000LL;
  v62 = 0;
  sub_E025C0(&v69, v64, &v66, &v62, a1, a7);
  if ( v62 )
  {
    v42 = *(_DWORD *)(a2 + 24);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(a2 + 8);
      v45 = 1;
      v46 = 0;
      v47 = v43 & (((0xBF58476D1CE4E5B9LL * v63) >> 31) ^ (484763065 * v63));
      v29 = (unsigned __int8 *)(v44 + 8LL * v47);
      v48 = *(_QWORD *)v29;
      if ( v63 == *(_QWORD *)v29 )
        goto LABEL_21;
      while ( v48 != -1 )
      {
        if ( v48 == -2 && !v46 )
          v46 = v29;
        v47 = v43 & (v45 + v47);
        v29 = (unsigned __int8 *)(v44 + 8LL * v47);
        v48 = *(_QWORD *)v29;
        if ( v63 == *(_QWORD *)v29 )
          goto LABEL_21;
        ++v45;
      }
      if ( !v46 )
        v46 = v29;
    }
    else
    {
      v46 = 0;
    }
    v49 = sub_D7AAE0(a2, &v63, v46);
    v50 = v63;
    *v49 = v63;
    v29 = *(unsigned __int8 **)(a2 + 40);
    if ( v29 == *(unsigned __int8 **)(a2 + 48) )
    {
      sub_9CA200(a2 + 32, v29, &v63);
    }
    else
    {
      if ( v29 )
      {
        *(_QWORD *)v29 = v50;
        v29 = *(unsigned __int8 **)(a2 + 40);
      }
      v29 += 8;
      *(_QWORD *)(a2 + 40) = v29;
    }
  }
LABEL_21:
  result = (__int64)v69;
  v38 = v69;
  v39 = &v69[2 * (unsigned int)v70];
  if ( v39 != v69 )
  {
    do
    {
      v40 = *v38;
      v29 = (unsigned __int8 *)v38[1];
      v38 += 2;
      result = sub_D7CB10(v40, v29, v63, a4, a6);
    }
    while ( v39 != v38 );
  }
  if ( v66 != v68 )
    result = _libc_free(v66, v29);
  v33 = v64[0];
  if ( (_BYTE *)v64[0] == v65 )
    goto LABEL_16;
LABEL_15:
  result = _libc_free(v33, v29);
LABEL_16:
  if ( v69 != (__int64 *)v71 )
    return _libc_free(v69, v29);
  return result;
}
