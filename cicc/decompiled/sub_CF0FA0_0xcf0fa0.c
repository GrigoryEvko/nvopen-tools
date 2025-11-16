// Function: sub_CF0FA0
// Address: 0xcf0fa0
//
__int64 __fastcall sub_CF0FA0(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // r12
  __int64 v6; // rbx
  _BYTE *v7; // r13
  _BYTE *v8; // r12
  __int64 v9; // rbx
  _QWORD *v10; // r14
  _QWORD *v11; // rbx
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // rbx
  _QWORD *v19; // r15
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  _QWORD *v22; // rbx
  __int64 v23; // rdx
  _BYTE *v24; // rbx
  _BYTE *v25; // r12
  __int64 v26; // r13
  _QWORD *v27; // r15
  _QWORD *v28; // r13
  _QWORD *v29; // rbx
  _QWORD *v30; // r12
  _BYTE *v31; // r12
  __int64 v32; // rcx
  __int64 *v33; // rbx
  int *v34; // rcx
  __int64 v35; // r15
  int *v36; // r13
  int v37; // eax
  unsigned __int8 v38; // [rsp+13h] [rbp-42Dh]
  unsigned int v39; // [rsp+14h] [rbp-42Ch]
  __int64 v40; // [rsp+18h] [rbp-428h]
  unsigned int v41; // [rsp+18h] [rbp-428h]
  __int64 v42; // [rsp+20h] [rbp-420h]
  __int64 v43; // [rsp+38h] [rbp-408h]
  __int64 v44; // [rsp+38h] [rbp-408h]
  int v45; // [rsp+40h] [rbp-400h]
  int v46; // [rsp+44h] [rbp-3FCh]
  char v47; // [rsp+48h] [rbp-3F8h]
  char v48; // [rsp+49h] [rbp-3F7h]
  char v49; // [rsp+4Ah] [rbp-3F6h]
  char v50; // [rsp+4Bh] [rbp-3F5h]
  int v51; // [rsp+4Ch] [rbp-3F4h]
  _BYTE *v52; // [rsp+50h] [rbp-3F0h] BYREF
  __int64 v53; // [rsp+58h] [rbp-3E8h]
  _BYTE v54[32]; // [rsp+60h] [rbp-3E0h] BYREF
  _BYTE *v55; // [rsp+80h] [rbp-3C0h] BYREF
  __int64 v56; // [rsp+88h] [rbp-3B8h]
  _BYTE v57[112]; // [rsp+90h] [rbp-3B0h] BYREF
  __int64 v58; // [rsp+100h] [rbp-340h] BYREF
  unsigned int v59; // [rsp+108h] [rbp-338h]
  char v60; // [rsp+110h] [rbp-330h] BYREF

  v38 = *(_BYTE *)(a1 + 96);
  if ( v38 )
    return v38;
  v2 = *(_QWORD *)(a1 + 56);
  sub_B428A0(&v58, (_BYTE *)v2, *(_QWORD *)(a1 + 64));
  if ( !v59 )
    goto LABEL_54;
  v43 = 0;
  v42 = 192LL * v59;
  while ( 1 )
  {
    v3 = v58 + v43;
    v45 = *(_DWORD *)(v58 + v43);
    v46 = *(_DWORD *)(v58 + v43 + 4);
    v47 = *(_BYTE *)(v58 + v43 + 8);
    v48 = *(_BYTE *)(v58 + v43 + 9);
    v49 = *(_BYTE *)(v58 + v43 + 10);
    v50 = *(_BYTE *)(v58 + v43 + 11);
    v51 = *(_DWORD *)(v58 + v43 + 12);
    v52 = v54;
    v53 = 0x100000000LL;
    if ( *(_DWORD *)(v58 + v43 + 24) )
    {
      v2 = v3 + 16;
      sub_CF0DB0((__int64)&v52, (__int64 *)(v3 + 16));
    }
    v55 = v57;
    v56 = 0x200000000LL;
    v4 = *(_DWORD *)(v3 + 72);
    if ( v4 )
    {
      if ( &v55 != (_BYTE **)(v3 + 64) )
        break;
    }
    if ( v45 == 2 )
      goto LABEL_9;
LABEL_71:
    v29 = v52;
    v30 = &v52[32 * (unsigned int)v53];
    if ( v52 != (_BYTE *)v30 )
    {
      do
      {
        v30 -= 4;
        if ( (_QWORD *)*v30 != v30 + 2 )
        {
          v2 = v30[2] + 1LL;
          j_j___libc_free_0(*v30, v2);
        }
      }
      while ( v29 != v30 );
      v30 = v52;
    }
    if ( v30 != (_QWORD *)v54 )
      _libc_free(v30, v2);
    v43 += 192;
    if ( v42 == v43 )
      goto LABEL_33;
  }
  v23 = v4;
  v31 = v57;
  v32 = v4;
  if ( v4 > 2 )
  {
    v39 = *(_DWORD *)(v3 + 72);
    v40 = v4;
    sub_B3C890((__int64)&v55, v4);
    v31 = v55;
    v32 = *(unsigned int *)(v3 + 72);
    v4 = v39;
    v23 = v40;
  }
  v33 = *(__int64 **)(v3 + 64);
  v2 = 7 * v32;
  v34 = (int *)&v33[7 * v32];
  if ( v33 != (__int64 *)v34 )
  {
    v41 = v4;
    v35 = v23;
    v36 = v34;
    do
    {
      if ( v31 )
      {
        v37 = *(_DWORD *)v33;
        *((_DWORD *)v31 + 4) = 0;
        *((_DWORD *)v31 + 5) = 1;
        *(_DWORD *)v31 = v37;
        *((_QWORD *)v31 + 1) = v31 + 24;
        if ( *((_DWORD *)v33 + 4) )
        {
          v2 = (__int64)(v33 + 1);
          sub_CF0DB0((__int64)(v31 + 8), v33 + 1);
        }
      }
      v33 += 7;
      v31 += 56;
    }
    while ( v36 != (int *)v33 );
    v4 = v41;
    v23 = v35;
  }
  LODWORD(v56) = v4;
  if ( v45 != 2 )
  {
LABEL_59:
    v24 = v55;
    v25 = &v55[56 * v23];
    if ( v55 != v25 )
    {
      do
      {
        v26 = *((unsigned int *)v25 - 10);
        v27 = (_QWORD *)*((_QWORD *)v25 - 6);
        v25 -= 56;
        v28 = &v27[4 * v26];
        if ( v27 != v28 )
        {
          do
          {
            v28 -= 4;
            if ( (_QWORD *)*v28 != v28 + 2 )
            {
              v2 = v28[2] + 1LL;
              j_j___libc_free_0(*v28, v2);
            }
          }
          while ( v27 != v28 );
          v27 = (_QWORD *)*((_QWORD *)v25 + 1);
        }
        if ( v27 != (_QWORD *)(v25 + 24) )
          _libc_free(v27, v2);
      }
      while ( v24 != v25 );
      v25 = v55;
    }
    if ( v25 != v57 )
      _libc_free(v25, v2);
    goto LABEL_71;
  }
LABEL_9:
  v5 = 0;
  v6 = 32LL * (unsigned int)v53;
  if ( !(_DWORD)v53 )
  {
LABEL_89:
    v23 = (unsigned int)v56;
    goto LABEL_59;
  }
  while ( 1 )
  {
    v2 = (__int64)"{memory}";
    if ( !(unsigned int)sub_2241AC0(&v52[v5], "{memory}") )
      break;
    v5 += 32;
    if ( v6 == v5 )
      goto LABEL_89;
  }
  v7 = v55;
  v8 = &v55[56 * (unsigned int)v56];
  if ( v55 != v8 )
  {
    do
    {
      v9 = *((unsigned int *)v8 - 10);
      v10 = (_QWORD *)*((_QWORD *)v8 - 6);
      v8 -= 56;
      v11 = &v10[4 * v9];
      if ( v10 != v11 )
      {
        do
        {
          v11 -= 4;
          if ( (_QWORD *)*v11 != v11 + 2 )
          {
            v2 = v11[2] + 1LL;
            j_j___libc_free_0(*v11, v2);
          }
        }
        while ( v10 != v11 );
        v10 = (_QWORD *)*((_QWORD *)v8 + 1);
      }
      if ( v10 != (_QWORD *)(v8 + 24) )
        _libc_free(v10, v2);
    }
    while ( v7 != v8 );
    v8 = v55;
  }
  if ( v8 != v57 )
    _libc_free(v8, v2);
  v12 = v52;
  v13 = &v52[32 * (unsigned int)v53];
  if ( v52 != (_BYTE *)v13 )
  {
    do
    {
      v13 -= 4;
      if ( (_QWORD *)*v13 != v13 + 2 )
      {
        v2 = v13[2] + 1LL;
        j_j___libc_free_0(*v13, v2);
      }
    }
    while ( v12 != v13 );
    v13 = v52;
  }
  if ( v13 != (_QWORD *)v54 )
    _libc_free(v13, v2);
  v38 = 1;
LABEL_33:
  v44 = v58;
  v14 = v58 + 192LL * v59;
  if ( v58 != v14 )
  {
    do
    {
      v15 = *(unsigned int *)(v14 - 120);
      v16 = *(_QWORD *)(v14 - 128);
      v14 -= 192;
      v17 = v16 + 56 * v15;
      if ( v16 != v17 )
      {
        do
        {
          v18 = *(unsigned int *)(v17 - 40);
          v19 = *(_QWORD **)(v17 - 48);
          v17 -= 56;
          v20 = &v19[4 * v18];
          if ( v19 != v20 )
          {
            do
            {
              v20 -= 4;
              if ( (_QWORD *)*v20 != v20 + 2 )
              {
                v2 = v20[2] + 1LL;
                j_j___libc_free_0(*v20, v2);
              }
            }
            while ( v19 != v20 );
            v19 = *(_QWORD **)(v17 + 8);
          }
          if ( v19 != (_QWORD *)(v17 + 24) )
            _libc_free(v19, v2);
        }
        while ( v16 != v17 );
        v16 = *(_QWORD *)(v14 + 64);
      }
      if ( v16 != v14 + 80 )
        _libc_free(v16, v2);
      v21 = *(_QWORD **)(v14 + 16);
      v22 = &v21[4 * *(unsigned int *)(v14 + 24)];
      if ( v21 != v22 )
      {
        do
        {
          v22 -= 4;
          if ( (_QWORD *)*v22 != v22 + 2 )
          {
            v2 = v22[2] + 1LL;
            j_j___libc_free_0(*v22, v2);
          }
        }
        while ( v21 != v22 );
        v21 = *(_QWORD **)(v14 + 16);
      }
      if ( v21 != (_QWORD *)(v14 + 32) )
        _libc_free(v21, v2);
    }
    while ( v44 != v14 );
LABEL_54:
    v14 = v58;
  }
  if ( (char *)v14 != &v60 )
    _libc_free(v14, v2);
  return v38;
}
