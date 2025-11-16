// Function: sub_1CCC8B0
// Address: 0x1ccc8b0
//
__int64 __fastcall sub_1CCC8B0(__int64 a1)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned int v7; // eax
  __int64 v8; // r12
  __int64 v9; // rbx
  _BYTE *v10; // r13
  unsigned __int64 v11; // r12
  __int64 v12; // rbx
  unsigned __int64 v13; // r14
  _QWORD *v14; // rbx
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  unsigned __int64 v17; // r13
  __int64 v18; // rdx
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // r12
  __int64 v21; // rbx
  unsigned __int64 v22; // r15
  _QWORD *v23; // rbx
  unsigned __int64 v24; // r12
  _QWORD *v25; // rbx
  _BYTE *v26; // rbx
  unsigned __int64 v27; // r12
  __int64 v28; // r13
  unsigned __int64 v29; // r15
  _QWORD *v30; // r13
  _QWORD *v31; // rbx
  _QWORD *v32; // r12
  __int64 v33; // rdx
  __int64 v34; // r15
  _BYTE *v35; // r12
  __int64 v36; // rcx
  __int64 *v37; // rbx
  __int64 v38; // r13
  int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-428h]
  unsigned __int8 v41; // [rsp+13h] [rbp-41Dh]
  unsigned int v42; // [rsp+14h] [rbp-41Ch]
  unsigned int v43; // [rsp+14h] [rbp-41Ch]
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

  v41 = *(_BYTE *)(a1 + 96);
  if ( v41 )
    return v41;
  sub_15F1410(&v59, *(_BYTE **)(a1 + 56), *(_QWORD *)(a1 + 64));
  if ( !v60 )
    goto LABEL_54;
  v44 = 0;
  v40 = 192LL * v60;
  while ( 1 )
  {
    v5 = v59 + v44;
    v46 = *(_DWORD *)(v59 + v44);
    v47 = *(_DWORD *)(v59 + v44 + 4);
    v48 = *(_BYTE *)(v59 + v44 + 8);
    v49 = *(_BYTE *)(v59 + v44 + 9);
    v50 = *(_BYTE *)(v59 + v44 + 10);
    v51 = *(_BYTE *)(v59 + v44 + 11);
    v52 = *(_DWORD *)(v59 + v44 + 12);
    v53 = v55;
    v54 = 0x100000000LL;
    v6 = *(unsigned int *)(v59 + v44 + 24);
    if ( (_DWORD)v6 )
      sub_1CCC070((__int64)&v53, (__int64 *)(v5 + 16), v6, v2, v3, v4);
    v56 = v58;
    v57 = 0x200000000LL;
    v7 = *(_DWORD *)(v5 + 72);
    if ( v7 )
    {
      v33 = v5 + 64;
      if ( &v56 != (_BYTE **)(v5 + 64) )
        break;
    }
    if ( v46 == 2 )
      goto LABEL_9;
LABEL_71:
    v31 = v53;
    v32 = &v53[32 * (unsigned int)v54];
    if ( v53 != (_BYTE *)v32 )
    {
      do
      {
        v32 -= 4;
        if ( (_QWORD *)*v32 != v32 + 2 )
          j_j___libc_free_0(*v32, v32[2] + 1LL);
      }
      while ( v31 != v32 );
      v32 = v53;
    }
    if ( v32 != (_QWORD *)v55 )
      _libc_free((unsigned __int64)v32);
    v44 += 192;
    if ( v40 == v44 )
      goto LABEL_33;
  }
  v34 = v7;
  if ( v7 > 2 )
  {
    v43 = *(_DWORD *)(v5 + 72);
    sub_15EB820((__int64)&v56, v7);
    v35 = v56;
    v36 = *(unsigned int *)(v5 + 72);
    v7 = v43;
  }
  else
  {
    v35 = v58;
    v36 = v7;
  }
  v37 = *(__int64 **)(v5 + 64);
  v2 = (__int64)&v37[7 * v36];
  if ( v37 != (__int64 *)v2 )
  {
    v42 = v7;
    v38 = v2;
    do
    {
      if ( v35 )
      {
        v39 = *(_DWORD *)v37;
        *((_DWORD *)v35 + 4) = 0;
        *((_DWORD *)v35 + 5) = 1;
        *(_DWORD *)v35 = v39;
        *((_QWORD *)v35 + 1) = v35 + 24;
        if ( *((_DWORD *)v37 + 4) )
          sub_1CCC070((__int64)(v35 + 8), v37 + 1, v33, v2, v3, v4);
      }
      v37 += 7;
      v35 += 56;
    }
    while ( (__int64 *)v38 != v37 );
    v7 = v42;
  }
  LODWORD(v57) = v7;
  if ( v46 != 2 )
  {
LABEL_59:
    v26 = v56;
    v27 = (unsigned __int64)&v56[56 * v34];
    if ( v56 != (_BYTE *)v27 )
    {
      do
      {
        v28 = *(unsigned int *)(v27 - 40);
        v29 = *(_QWORD *)(v27 - 48);
        v27 -= 56LL;
        v30 = (_QWORD *)(v29 + 32 * v28);
        if ( (_QWORD *)v29 != v30 )
        {
          do
          {
            v30 -= 4;
            if ( (_QWORD *)*v30 != v30 + 2 )
              j_j___libc_free_0(*v30, v30[2] + 1LL);
          }
          while ( (_QWORD *)v29 != v30 );
          v29 = *(_QWORD *)(v27 + 8);
        }
        if ( v29 != v27 + 24 )
          _libc_free(v29);
      }
      while ( v26 != (_BYTE *)v27 );
      v27 = (unsigned __int64)v56;
    }
    if ( (_BYTE *)v27 != v58 )
      _libc_free(v27);
    goto LABEL_71;
  }
LABEL_9:
  v8 = 0;
  v9 = 32LL * (unsigned int)v54;
  if ( !(_DWORD)v54 )
  {
LABEL_90:
    v34 = (unsigned int)v57;
    goto LABEL_59;
  }
  while ( (unsigned int)sub_2241AC0(&v53[v8], "{memory}") )
  {
    v8 += 32;
    if ( v9 == v8 )
      goto LABEL_90;
  }
  v10 = v56;
  v11 = (unsigned __int64)&v56[56 * (unsigned int)v57];
  if ( v56 != (_BYTE *)v11 )
  {
    do
    {
      v12 = *(unsigned int *)(v11 - 40);
      v13 = *(_QWORD *)(v11 - 48);
      v11 -= 56LL;
      v14 = (_QWORD *)(v13 + 32 * v12);
      if ( (_QWORD *)v13 != v14 )
      {
        do
        {
          v14 -= 4;
          if ( (_QWORD *)*v14 != v14 + 2 )
            j_j___libc_free_0(*v14, v14[2] + 1LL);
        }
        while ( (_QWORD *)v13 != v14 );
        v13 = *(_QWORD *)(v11 + 8);
      }
      if ( v13 != v11 + 24 )
        _libc_free(v13);
    }
    while ( v10 != (_BYTE *)v11 );
    v11 = (unsigned __int64)v56;
  }
  if ( (_BYTE *)v11 != v58 )
    _libc_free(v11);
  v15 = v53;
  v16 = &v53[32 * (unsigned int)v54];
  if ( v53 != (_BYTE *)v16 )
  {
    do
    {
      v16 -= 4;
      if ( (_QWORD *)*v16 != v16 + 2 )
        j_j___libc_free_0(*v16, v16[2] + 1LL);
    }
    while ( v15 != v16 );
    v16 = v53;
  }
  if ( v16 != (_QWORD *)v55 )
    _libc_free((unsigned __int64)v16);
  v41 = 1;
LABEL_33:
  v45 = v59;
  v17 = v59 + 192LL * v60;
  if ( v59 != v17 )
  {
    do
    {
      v18 = *(unsigned int *)(v17 - 120);
      v19 = *(_QWORD *)(v17 - 128);
      v17 -= 192LL;
      v20 = v19 + 56 * v18;
      if ( v19 != v20 )
      {
        do
        {
          v21 = *(unsigned int *)(v20 - 40);
          v22 = *(_QWORD *)(v20 - 48);
          v20 -= 56LL;
          v23 = (_QWORD *)(v22 + 32 * v21);
          if ( (_QWORD *)v22 != v23 )
          {
            do
            {
              v23 -= 4;
              if ( (_QWORD *)*v23 != v23 + 2 )
                j_j___libc_free_0(*v23, v23[2] + 1LL);
            }
            while ( (_QWORD *)v22 != v23 );
            v22 = *(_QWORD *)(v20 + 8);
          }
          if ( v22 != v20 + 24 )
            _libc_free(v22);
        }
        while ( v19 != v20 );
        v19 = *(_QWORD *)(v17 + 64);
      }
      if ( v19 != v17 + 80 )
        _libc_free(v19);
      v24 = *(_QWORD *)(v17 + 16);
      v25 = (_QWORD *)(v24 + 32LL * *(unsigned int *)(v17 + 24));
      if ( (_QWORD *)v24 != v25 )
      {
        do
        {
          v25 -= 4;
          if ( (_QWORD *)*v25 != v25 + 2 )
            j_j___libc_free_0(*v25, v25[2] + 1LL);
        }
        while ( (_QWORD *)v24 != v25 );
        v24 = *(_QWORD *)(v17 + 16);
      }
      if ( v24 != v17 + 32 )
        _libc_free(v24);
    }
    while ( v45 != v17 );
LABEL_54:
    v17 = v59;
  }
  if ( (char *)v17 != &v61 )
    _libc_free(v17);
  return v41;
}
