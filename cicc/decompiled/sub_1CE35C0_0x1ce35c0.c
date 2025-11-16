// Function: sub_1CE35C0
// Address: 0x1ce35c0
//
__int64 __fastcall sub_1CE35C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        _QWORD *a7,
        int a8)
{
  unsigned int v11; // r12d
  __int64 v13; // rdi
  _BYTE *v14; // rsi
  unsigned int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rax
  int v20; // r14d
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // r14
  __int64 v25; // r12
  _BYTE *v26; // r10
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned int v29; // edx
  _QWORD *v30; // rcx
  _BYTE *v31; // rdi
  _BYTE *v32; // rsi
  _BYTE *v33; // rcx
  __int64 v34; // r8
  __int64 v35; // rdi
  _QWORD *v36; // rdx
  unsigned __int64 v37; // r11
  int v38; // r10d
  unsigned __int64 v39; // rsi
  __int64 v40; // rax
  int v41; // ecx
  unsigned __int8 v42; // al
  __int64 v43; // rax
  __int64 v44; // rax
  _BYTE *v45; // rsi
  _QWORD *v46; // rbx
  _QWORD *v47; // r12
  _QWORD *v48; // r13
  _QWORD *v49; // rdx
  __int64 v50; // rdi
  int v51; // eax
  int v52; // r9d
  unsigned __int8 v53; // [rsp+3h] [rbp-9Dh]
  int v54; // [rsp+8h] [rbp-98h]
  __int64 v55; // [rsp+8h] [rbp-98h]
  int v56; // [rsp+10h] [rbp-90h]
  int v58; // [rsp+20h] [rbp-80h]
  __int64 v59; // [rsp+20h] [rbp-80h]
  _BYTE *v60; // [rsp+20h] [rbp-80h]
  __int64 v62; // [rsp+30h] [rbp-70h]
  __int64 v63; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v64[4]; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int8 v65; // [rsp+60h] [rbp-40h]

  v63 = a2;
  v56 = a4;
  sub_1A64820((__int64)v64, a4, &v63);
  v11 = v65;
  if ( !v65 )
    return 1;
  if ( a8 == 51 )
    return 0;
  v13 = v63;
  v14 = (_BYTE *)a6[1];
  v64[0] = v63;
  if ( v14 == (_BYTE *)a6[2] )
  {
    sub_12879C0((__int64)a6, v14, v64);
    v13 = v63;
  }
  else
  {
    if ( v14 )
    {
      *(_QWORD *)v14 = v63;
      v14 = (_BYTE *)a6[1];
      v13 = v63;
    }
    a6[1] = v14 + 8;
  }
  v15 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
  if ( *(_BYTE *)(v13 + 16) != 78 )
    goto LABEL_17;
  if ( *(char *)(v13 + 23) < 0 )
  {
    v54 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
    v16 = sub_1648A40(v13);
    v15 = v54;
    v18 = v16 + v17;
    if ( *(char *)(v13 + 23) >= 0 )
    {
      if ( (unsigned int)(v18 >> 4) )
        goto LABEL_84;
    }
    else
    {
      v19 = sub_1648A40(v13);
      v15 = v54;
      if ( (unsigned int)((v18 - v19) >> 4) )
      {
        if ( *(char *)(v13 + 23) < 0 )
        {
          v20 = *(_DWORD *)(sub_1648A40(v13) + 8);
          if ( *(char *)(v13 + 23) >= 0 )
            BUG();
          v21 = sub_1648A40(v13);
          v15 = v54;
          v23 = *(_DWORD *)(v21 + v22 - 4) - v20;
          goto LABEL_16;
        }
LABEL_84:
        BUG();
      }
    }
  }
  v23 = 0;
LABEL_16:
  v15 = v15 - 1 - v23;
LABEL_17:
  v24 = 0;
  v62 = v15;
  if ( !v15 )
    goto LABEL_45;
  v53 = v11;
  v25 = a5;
  do
  {
    if ( *(_BYTE *)(v63 + 16) == 78 )
    {
      v26 = *(_BYTE **)(v63 + 24 * (v24 - (*(_DWORD *)(v63 + 20) & 0xFFFFFFF)));
    }
    else
    {
      if ( (*(_BYTE *)(v63 + 23) & 0x40) != 0 )
        v44 = *(_QWORD *)(v63 - 8);
      else
        v44 = v63 - 24LL * (*(_DWORD *)(v63 + 20) & 0xFFFFFFF);
      v26 = *(_BYTE **)(v44 + 24 * v24);
    }
    v27 = *(unsigned int *)(a3 + 24);
    v64[0] = v26;
    if ( (_DWORD)v27 )
    {
      v28 = *(_QWORD *)(a3 + 8);
      v29 = (v27 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v30 = (_QWORD *)(v28 + 8LL * v29);
      v31 = (_BYTE *)*v30;
      if ( v26 == (_BYTE *)*v30 )
      {
LABEL_23:
        if ( v30 != (_QWORD *)(v28 + 8 * v27) )
        {
          if ( (unsigned __int8)sub_1CE34D0(a1, (__int64)v26, v25) )
            goto LABEL_43;
          goto LABEL_25;
        }
      }
      else
      {
        v41 = 1;
        while ( v31 != (_BYTE *)-8LL )
        {
          v52 = v41 + 1;
          v29 = (v27 - 1) & (v41 + v29);
          v30 = (_QWORD *)(v28 + 8LL * v29);
          v31 = (_BYTE *)*v30;
          if ( v26 == (_BYTE *)*v30 )
            goto LABEL_23;
          v41 = v52;
        }
      }
    }
    v42 = v26[16];
    if ( v42 <= 0x17u )
      goto LABEL_43;
    if ( v42 == 54 )
    {
      v43 = **((_QWORD **)v26 - 3);
      if ( *(_BYTE *)(v43 + 8) == 16 )
        v43 = **(_QWORD **)(v43 + 16);
      if ( *(_DWORD *)(v43 + 8) >> 8 == 101 )
        goto LABEL_43;
    }
    else if ( v42 == 86 )
    {
      v50 = *((_QWORD *)v26 - 3);
      if ( *(_BYTE *)(v50 + 16) == 54 )
      {
        v60 = v26;
        v51 = sub_1776BC0(v50);
        v26 = v60;
        if ( v51 == 101 )
          goto LABEL_43;
      }
    }
    v58 = (int)v26;
    if ( sub_1CD06C0(v26, 0) )
    {
      if ( !(unsigned __int8)sub_1CE35C0(a1, v58, a3, v56, v25, (_DWORD)a6, (__int64)a7, a8 + 1) )
        return 0;
      goto LABEL_43;
    }
LABEL_25:
    v32 = (_BYTE *)a6[1];
    if ( v32 == (_BYTE *)a6[2] )
    {
      sub_1287830((__int64)a6, v32, v64);
      v33 = (_BYTE *)a6[1];
    }
    else
    {
      if ( v32 )
      {
        *(_QWORD *)v32 = v64[0];
        v32 = (_BYTE *)a6[1];
      }
      v33 = v32 + 8;
      a6[1] = v32 + 8;
    }
    v34 = a7[1];
    v35 = *a7;
    v36 = (_QWORD *)*a6;
    if ( v34 == *a7 )
    {
      if ( v33 == (_BYTE *)v36 )
        goto LABEL_42;
      v59 = a3;
      v45 = (_BYTE *)a7[1];
      v46 = v33;
      v55 = v25;
      v47 = a7;
      v48 = (_QWORD *)*a6;
      while ( 1 )
      {
        if ( (_BYTE *)v47[2] == v45 )
        {
          v49 = v48++;
          sub_1287830((__int64)v47, v45, v49);
          if ( v48 == v46 )
            goto LABEL_69;
        }
        else
        {
          if ( v45 )
          {
            *(_QWORD *)v45 = *v48;
            v45 = (_BYTE *)v47[1];
          }
          ++v48;
          v47[1] = v45 + 8;
          if ( v48 == v46 )
          {
LABEL_69:
            a7 = v47;
            a3 = v59;
            v33 = (_BYTE *)a6[1];
            v25 = v55;
            goto LABEL_42;
          }
        }
        v45 = (_BYTE *)v47[1];
      }
    }
    v37 = (v34 - v35) >> 3;
    v38 = (v33 - (_BYTE *)v36) >> 3;
    if ( !v38 || !(_DWORD)v37 )
      return 0;
    LODWORD(v39) = 0;
    while ( *(_QWORD *)(v35 + 8LL * (unsigned int)v39) == v36[(unsigned int)v39] )
    {
      v39 = (unsigned int)(v39 + 1);
      if ( (unsigned int)((v34 - v35) >> 3) == (_DWORD)v39 || v38 == (_DWORD)v39 )
        goto LABEL_37;
    }
    v39 = (unsigned int)(v39 - 1);
LABEL_37:
    if ( !(_DWORD)v39 )
      return 0;
    if ( v39 > v37 )
    {
      sub_185FB80((__int64)a7, v39 - v37);
      v33 = (_BYTE *)a6[1];
    }
    else if ( v39 < v37 )
    {
      v40 = v35 + 8 * v39;
      if ( v34 != v40 )
      {
        a7[1] = v40;
        v33 = (_BYTE *)a6[1];
      }
    }
LABEL_42:
    a6[1] = v33 - 8;
LABEL_43:
    ++v24;
  }
  while ( v24 != v62 );
  v11 = v53;
LABEL_45:
  a6[1] -= 8LL;
  return v11;
}
