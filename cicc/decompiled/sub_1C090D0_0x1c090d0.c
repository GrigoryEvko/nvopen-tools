// Function: sub_1C090D0
// Address: 0x1c090d0
//
__int64 __fastcall sub_1C090D0(__int64 a1)
{
  __int64 v1; // rcx
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // r12
  _BYTE *v9; // rbx
  unsigned __int64 v10; // r12
  __int64 v11; // r13
  unsigned __int64 v12; // r14
  _QWORD *v13; // r13
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  unsigned __int64 v16; // r14
  __int64 v17; // rdx
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // r12
  __int64 v20; // rbx
  unsigned __int64 v21; // r15
  _QWORD *v22; // rbx
  unsigned __int64 v23; // r12
  _QWORD *v24; // rbx
  __int64 v26; // r15
  _BYTE *v27; // rbx
  unsigned __int64 v28; // r12
  __int64 v29; // r14
  unsigned __int64 v30; // r15
  _QWORD *v31; // r14
  _QWORD *v32; // rbx
  _QWORD *v33; // r12
  _BYTE *v34; // r12
  __int64 v35; // rcx
  __int64 *v36; // rbx
  __int64 v37; // r14
  int v38; // r15d
  int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-428h]
  unsigned int v41; // [rsp+10h] [rbp-420h]
  __int64 v42; // [rsp+10h] [rbp-420h]
  unsigned int v43; // [rsp+10h] [rbp-420h]
  __int64 v44; // [rsp+28h] [rbp-408h]
  unsigned __int64 v45; // [rsp+28h] [rbp-408h]
  int v46; // [rsp+30h] [rbp-400h]
  int v47; // [rsp+34h] [rbp-3FCh]
  char v48; // [rsp+38h] [rbp-3F8h]
  char v49; // [rsp+39h] [rbp-3F7h]
  char v50; // [rsp+3Ah] [rbp-3F6h]
  char v51; // [rsp+3Bh] [rbp-3F5h]
  int v52; // [rsp+3Ch] [rbp-3F4h]
  _BYTE *v53; // [rsp+40h] [rbp-3F0h] BYREF
  __int64 v54; // [rsp+48h] [rbp-3E8h]
  _BYTE v55[32]; // [rsp+50h] [rbp-3E0h] BYREF
  _BYTE *v56; // [rsp+70h] [rbp-3C0h] BYREF
  __int64 v57; // [rsp+78h] [rbp-3B8h]
  _BYTE v58[112]; // [rsp+80h] [rbp-3B0h] BYREF
  unsigned __int64 v59; // [rsp+F0h] [rbp-340h] BYREF
  unsigned int v60; // [rsp+F8h] [rbp-338h]
  char v61; // [rsp+100h] [rbp-330h] BYREF

  sub_15F1410(&v59, *(_BYTE **)(a1 + 56), *(_QWORD *)(a1 + 64));
  v41 = v60;
  if ( !v60 )
    goto LABEL_55;
  v44 = 0;
  v40 = 192LL * v60;
  while ( 1 )
  {
    v4 = v59 + v44;
    v46 = *(_DWORD *)(v59 + v44);
    v47 = *(_DWORD *)(v59 + v44 + 4);
    v48 = *(_BYTE *)(v59 + v44 + 8);
    v49 = *(_BYTE *)(v59 + v44 + 9);
    v50 = *(_BYTE *)(v59 + v44 + 10);
    v51 = *(_BYTE *)(v59 + v44 + 11);
    v52 = *(_DWORD *)(v59 + v44 + 12);
    v53 = v55;
    v54 = 0x100000000LL;
    v5 = *(unsigned int *)(v59 + v44 + 24);
    if ( (_DWORD)v5 )
      sub_1C08E50((__int64)&v53, (__int64 *)(v4 + 16), v5, v1, v2, v3);
    v56 = v58;
    v57 = 0x200000000LL;
    v6 = *(unsigned int *)(v4 + 72);
    if ( (_DWORD)v6 )
    {
      if ( &v56 != (_BYTE **)(v4 + 64) )
        break;
    }
    if ( v46 == 2 )
      goto LABEL_7;
LABEL_73:
    v32 = v53;
    v33 = &v53[32 * (unsigned int)v54];
    if ( v53 != (_BYTE *)v33 )
    {
      do
      {
        v33 -= 4;
        if ( (_QWORD *)*v33 != v33 + 2 )
          j_j___libc_free_0(*v33, v33[2] + 1LL);
      }
      while ( v32 != v33 );
      v33 = v53;
    }
    if ( v33 != (_QWORD *)v55 )
      _libc_free((unsigned __int64)v33);
    v44 += 192;
    if ( v40 == v44 )
    {
      v41 = 0;
      goto LABEL_34;
    }
  }
  v26 = (unsigned int)v6;
  if ( (unsigned int)v6 > 2 )
  {
    v43 = *(_DWORD *)(v4 + 72);
    sub_15EB820((__int64)&v56, (unsigned int)v6);
    v34 = v56;
    v35 = *(unsigned int *)(v4 + 72);
    v6 = v43;
  }
  else
  {
    v34 = v58;
    v35 = (unsigned int)v6;
  }
  v36 = *(__int64 **)(v4 + 64);
  v1 = (__int64)&v36[7 * v35];
  if ( v36 != (__int64 *)v1 )
  {
    v42 = v26;
    v37 = v1;
    v38 = v6;
    do
    {
      if ( v34 )
      {
        v39 = *(_DWORD *)v36;
        *((_DWORD *)v34 + 4) = 0;
        *((_DWORD *)v34 + 5) = 1;
        *(_DWORD *)v34 = v39;
        *((_QWORD *)v34 + 1) = v34 + 24;
        if ( *((_DWORD *)v36 + 4) )
          sub_1C08E50((__int64)(v34 + 8), v36 + 1, v6, v1, v2, v3);
      }
      v36 += 7;
      v34 += 56;
    }
    while ( (__int64 *)v37 != v36 );
    LODWORD(v6) = v38;
    v26 = v42;
  }
  LODWORD(v57) = v6;
  if ( v46 != 2 )
  {
LABEL_61:
    v27 = v56;
    v28 = (unsigned __int64)&v56[56 * v26];
    if ( v56 != (_BYTE *)v28 )
    {
      do
      {
        v29 = *(unsigned int *)(v28 - 40);
        v30 = *(_QWORD *)(v28 - 48);
        v28 -= 56LL;
        v31 = (_QWORD *)(v30 + 32 * v29);
        if ( (_QWORD *)v30 != v31 )
        {
          do
          {
            v31 -= 4;
            if ( (_QWORD *)*v31 != v31 + 2 )
              j_j___libc_free_0(*v31, v31[2] + 1LL);
          }
          while ( (_QWORD *)v30 != v31 );
          v30 = *(_QWORD *)(v28 + 8);
        }
        if ( v30 != v28 + 24 )
          _libc_free(v30);
      }
      while ( v27 != (_BYTE *)v28 );
      v28 = (unsigned __int64)v56;
    }
    if ( (_BYTE *)v28 != v58 )
      _libc_free(v28);
    goto LABEL_73;
  }
LABEL_7:
  v7 = 0;
  v8 = 32LL * (unsigned int)v54;
  if ( !(_DWORD)v54 )
  {
LABEL_92:
    v26 = (unsigned int)v57;
    goto LABEL_61;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(&v53[v7], "{thvar}") )
    {
      v41 = 7;
      goto LABEL_15;
    }
    if ( !(unsigned int)sub_2241AC0(&v53[v7], "{xthvar}") )
      break;
    if ( !(unsigned int)sub_2241AC0(&v53[v7], "{ythvar}") )
    {
      v41 = 2;
      goto LABEL_15;
    }
    if ( !(unsigned int)sub_2241AC0(&v53[v7], "{zthvar}") )
    {
      v41 = 4;
      goto LABEL_15;
    }
    v7 += 32;
    if ( v8 == v7 )
      goto LABEL_92;
  }
  v41 = 1;
LABEL_15:
  v9 = v56;
  v10 = (unsigned __int64)&v56[56 * (unsigned int)v57];
  if ( v56 != (_BYTE *)v10 )
  {
    do
    {
      v11 = *(unsigned int *)(v10 - 40);
      v12 = *(_QWORD *)(v10 - 48);
      v10 -= 56LL;
      v13 = (_QWORD *)(v12 + 32 * v11);
      if ( (_QWORD *)v12 != v13 )
      {
        do
        {
          v13 -= 4;
          if ( (_QWORD *)*v13 != v13 + 2 )
            j_j___libc_free_0(*v13, v13[2] + 1LL);
        }
        while ( (_QWORD *)v12 != v13 );
        v12 = *(_QWORD *)(v10 + 8);
      }
      if ( v12 != v10 + 24 )
        _libc_free(v12);
    }
    while ( v9 != (_BYTE *)v10 );
    v10 = (unsigned __int64)v56;
  }
  if ( (_BYTE *)v10 != v58 )
    _libc_free(v10);
  v14 = v53;
  v15 = &v53[32 * (unsigned int)v54];
  if ( v53 != (_BYTE *)v15 )
  {
    do
    {
      v15 -= 4;
      if ( (_QWORD *)*v15 != v15 + 2 )
        j_j___libc_free_0(*v15, v15[2] + 1LL);
    }
    while ( v14 != v15 );
    v15 = v53;
  }
  if ( v15 != (_QWORD *)v55 )
    _libc_free((unsigned __int64)v15);
LABEL_34:
  v45 = v59;
  v16 = v59 + 192LL * v60;
  if ( v59 != v16 )
  {
    do
    {
      v17 = *(unsigned int *)(v16 - 120);
      v18 = *(_QWORD *)(v16 - 128);
      v16 -= 192LL;
      v19 = v18 + 56 * v17;
      if ( v18 != v19 )
      {
        do
        {
          v20 = *(unsigned int *)(v19 - 40);
          v21 = *(_QWORD *)(v19 - 48);
          v19 -= 56LL;
          v22 = (_QWORD *)(v21 + 32 * v20);
          if ( (_QWORD *)v21 != v22 )
          {
            do
            {
              v22 -= 4;
              if ( (_QWORD *)*v22 != v22 + 2 )
                j_j___libc_free_0(*v22, v22[2] + 1LL);
            }
            while ( (_QWORD *)v21 != v22 );
            v21 = *(_QWORD *)(v19 + 8);
          }
          if ( v21 != v19 + 24 )
            _libc_free(v21);
        }
        while ( v18 != v19 );
        v18 = *(_QWORD *)(v16 + 64);
      }
      if ( v18 != v16 + 80 )
        _libc_free(v18);
      v23 = *(_QWORD *)(v16 + 16);
      v24 = (_QWORD *)(v23 + 32LL * *(unsigned int *)(v16 + 24));
      if ( (_QWORD *)v23 != v24 )
      {
        do
        {
          v24 -= 4;
          if ( (_QWORD *)*v24 != v24 + 2 )
            j_j___libc_free_0(*v24, v24[2] + 1LL);
        }
        while ( (_QWORD *)v23 != v24 );
        v23 = *(_QWORD *)(v16 + 16);
      }
      if ( v23 != v16 + 32 )
        _libc_free(v23);
    }
    while ( v45 != v16 );
LABEL_55:
    v16 = v59;
  }
  if ( (char *)v16 != &v61 )
    _libc_free(v16);
  return v41;
}
