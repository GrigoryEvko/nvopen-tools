// Function: sub_2B57A50
// Address: 0x2b57a50
//
char __fastcall sub_2B57A50(__int64 *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rsi
  char v6; // cl
  __int64 v7; // rdi
  int v8; // eax
  unsigned int v9; // edx
  __int64 *v10; // r12
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rbx
  __int64 v17; // rax
  _DWORD *v18; // rdx
  _DWORD *v19; // rcx
  char v20; // al
  _QWORD *v21; // r15
  __int64 v22; // rdi
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 *v27; // rdi
  unsigned int *v28; // r8
  __int64 *v29; // r12
  __int64 v30; // rsi
  _QWORD *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // r14
  int v37; // edx
  __int64 v38; // rax
  int v39; // edx
  bool v40; // sf
  bool v41; // of
  char v42; // r14
  __int64 v43; // rax
  __int64 v44; // rsi
  int v45; // edx
  unsigned int v46; // eax
  _QWORD *v47; // rcx
  int v48; // edi
  __int64 *v49; // rbx
  __int64 v50; // rdx
  __int64 v51; // rsi
  int v52; // edx
  __int64 v53; // rdi
  __int64 v54; // rdx
  int v55; // r9d
  __int64 *v57; // [rsp+18h] [rbp-158h]
  __int64 v58; // [rsp+18h] [rbp-158h]
  int v59; // [rsp+20h] [rbp-150h]
  unsigned __int64 v60; // [rsp+28h] [rbp-148h]
  unsigned int *v61; // [rsp+28h] [rbp-148h]
  unsigned __int64 v63; // [rsp+38h] [rbp-138h]
  char v64; // [rsp+38h] [rbp-138h]
  __int64 *v65; // [rsp+38h] [rbp-138h]
  _BOOL4 v66; // [rsp+4Ch] [rbp-124h] BYREF
  __int64 v67; // [rsp+50h] [rbp-120h] BYREF
  _QWORD *v68; // [rsp+58h] [rbp-118h] BYREF
  __int64 *v69; // [rsp+60h] [rbp-110h] BYREF
  __int64 v70; // [rsp+68h] [rbp-108h]
  _BYTE v71[48]; // [rsp+70h] [rbp-100h] BYREF
  _BYTE v72[24]; // [rsp+A0h] [rbp-D0h] BYREF
  char *v73; // [rsp+B8h] [rbp-B8h]
  char v74; // [rsp+C8h] [rbp-A8h] BYREF
  char *v75; // [rsp+E8h] [rbp-88h]
  char v76; // [rsp+F8h] [rbp-78h] BYREF

  v5 = *a1;
  v6 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v6 )
  {
    v7 = v5 + 16;
    v8 = 3;
  }
  else
  {
    v33 = *(unsigned int *)(v5 + 24);
    v7 = *(_QWORD *)(v5 + 16);
    if ( !(_DWORD)v33 )
      goto LABEL_84;
    v8 = v33 - 1;
  }
  v9 = v8 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a3 == *v10 )
    goto LABEL_4;
  v55 = 1;
  while ( v11 != -4096 )
  {
    v9 = v8 & (v55 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a3 == *v10 )
      goto LABEL_4;
    ++v55;
  }
  if ( v6 )
  {
    v54 = 64;
    goto LABEL_85;
  }
  v33 = *(unsigned int *)(v5 + 24);
LABEL_84:
  v54 = 16 * v33;
LABEL_85:
  v10 = (__int64 *)(v7 + v54);
LABEL_4:
  v12 = 64;
  if ( !v6 )
    v12 = 16LL * *(unsigned int *)(v5 + 24);
  if ( v10 == (__int64 *)(v7 + v12) )
  {
    if ( a3 == a2 || sub_B445A0(a3, a2) )
    {
      LOBYTE(v32) = 1;
      return v32;
    }
    goto LABEL_9;
  }
  v13 = v10[1];
  a3 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 == (v13 & 0xFFFFFFFFFFFFFFF8LL) )
    return (v13 >> 2) & 1;
  if ( sub_B445A0(v10[1] & 0xFFFFFFFFFFFFFFF8LL, a2) )
  {
    v13 = v10[1];
    return (v13 >> 2) & 1;
  }
LABEL_9:
  v16 = (_QWORD *)(a3 + 24);
  v17 = *(_QWORD *)(a2 + 24);
  v69 = (__int64 *)v71;
  v70 = 0x600000000LL;
  v63 = v17 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v16 == (_QWORD *)(v17 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v19 = (_DWORD *)a1[1];
    v18 = (_DWORD *)a1[2];
    v27 = (__int64 *)v71;
LABEL_33:
    LOBYTE(v32) = *v19 <= *v18;
    goto LABEL_34;
  }
  while ( 1 )
  {
    v18 = (_DWORD *)a1[2];
    v19 = (_DWORD *)a1[1];
    if ( *v19 > *v18 )
      goto LABEL_25;
    if ( !v16 )
      BUG();
    v20 = *((_BYTE *)v16 - 24);
    v21 = v16 - 3;
    if ( v20 != 85 )
      break;
    v34 = *(v16 - 7);
    if ( !v34 || *(_BYTE *)v34 || *(_QWORD *)(v34 + 24) != v16[7] || (*(_BYTE *)(v34 + 33) & 0x20) == 0 )
      goto LABEL_56;
    v35 = *(_DWORD *)(v34 + 36);
    if ( v35 > 0xD3 )
    {
      if ( v35 == 324 )
        goto LABEL_15;
      if ( v35 > 0x144 )
      {
        if ( v35 == 376 )
          goto LABEL_15;
      }
      else if ( v35 == 282 || v35 - 291 <= 1 )
      {
        goto LABEL_15;
      }
    }
    else if ( v35 > 0x9A )
    {
      v53 = 0x186000000000001LL;
      if ( _bittest64(&v53, v35 - 155) )
        goto LABEL_15;
    }
    else if ( v35 == 11 || v35 - 68 <= 3 )
    {
      goto LABEL_15;
    }
    v36 = a1[3];
    v60 = v60 & 0xFFFFFFFF00000000LL | 1;
    sub_DF86E0((__int64)v72, v35, (unsigned __int8 *)v16 - 24, 0, v60, 0, 0);
    v58 = sub_DFD690(*(_QWORD *)(*(_QWORD *)v36 + 3296LL), (__int64)v72);
    v59 = v37;
    v38 = sub_DFD7B0(*(_QWORD *)(*(_QWORD *)v36 + 3296LL));
    v41 = __OFSUB__(v59, v39);
    v40 = v59 - v39 < 0;
    if ( v59 == v39 )
    {
      v41 = __OFSUB__(v58, v38);
      v40 = v58 - v38 < 0;
    }
    v42 = v40 ^ v41;
    if ( v75 != &v76 )
      _libc_free((unsigned __int64)v75);
    if ( v73 != &v74 )
      _libc_free((unsigned __int64)v73);
    if ( !v42 )
      goto LABEL_56;
LABEL_15:
    v22 = a1[5];
    if ( *(_BYTE *)(v22 + 28) )
    {
      v23 = *(_QWORD **)(v22 + 8);
      v24 = &v23[*(unsigned int *)(v22 + 20)];
      if ( v23 != v24 )
      {
        while ( v21 != (_QWORD *)*v23 )
        {
          if ( v24 == ++v23 )
            goto LABEL_23;
        }
LABEL_20:
        v25 = (unsigned int)v70;
        v26 = (unsigned int)v70 + 1LL;
        if ( v26 > HIDWORD(v70) )
        {
          sub_C8D5F0((__int64)&v69, v71, v26, 8u, v14, v15);
          v25 = (unsigned int)v70;
        }
        v69[v25] = (__int64)v21;
        LODWORD(v70) = v70 + 1;
      }
    }
    else if ( sub_C8CA60(v22, (__int64)(v16 - 3)) )
    {
      goto LABEL_20;
    }
LABEL_23:
    v16 = (_QWORD *)(*v16 & 0xFFFFFFFFFFFFFFF8LL);
    ++*(_DWORD *)a1[1];
    if ( (_QWORD *)v63 == v16 )
    {
      v19 = (_DWORD *)a1[1];
      v18 = (_DWORD *)a1[2];
LABEL_25:
      v27 = v69;
      v57 = &v69[(unsigned int)v70];
      if ( v57 != v69 )
      {
        v28 = (unsigned int *)&v66;
        v29 = v69;
        while ( 1 )
        {
          v67 = *v29;
          v30 = *a1;
          v66 = *v19 <= *v18;
          v31 = (_QWORD *)a2;
          if ( (_QWORD *)v63 != v16 )
          {
            v31 = v16 - 3;
            if ( !v16 )
              v31 = 0;
          }
          ++v29;
          v68 = v31;
          v61 = v28;
          sub_2B57670((__int64)v72, v30, &v67, &v68, v28);
          v28 = v61;
          if ( v57 == v29 )
            break;
          v19 = (_DWORD *)a1[1];
          v18 = (_DWORD *)a1[2];
        }
        v19 = (_DWORD *)a1[1];
        v18 = (_DWORD *)a1[2];
        v27 = v69;
      }
      goto LABEL_33;
    }
  }
  if ( v20 != 34 && v20 != 40 )
    goto LABEL_15;
LABEL_56:
  v43 = a1[4];
  if ( (*(_BYTE *)(v43 + 88) & 1) != 0 )
  {
    v44 = v43 + 96;
    v45 = 3;
  }
  else
  {
    v52 = *(_DWORD *)(v43 + 104);
    v44 = *(_QWORD *)(v43 + 96);
    if ( !v52 )
      goto LABEL_61;
    v45 = v52 - 1;
  }
  v46 = v45 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v47 = *(_QWORD **)(v44 + 72LL * v46);
  if ( v21 == v47 )
    goto LABEL_15;
  v48 = 1;
  while ( v47 != (_QWORD *)-4096LL )
  {
    v14 = (unsigned int)(v48 + 1);
    v46 = v45 & (v48 + v46);
    v47 = *(_QWORD **)(v44 + 72LL * v46);
    if ( v21 == v47 )
      goto LABEL_15;
    ++v48;
  }
LABEL_61:
  v27 = v69;
  v65 = &v69[(unsigned int)v70];
  if ( v65 != v69 )
  {
    v49 = v69;
    do
    {
      v50 = *v49;
      v51 = *a1;
      ++v49;
      v66 = 0;
      v67 = v50;
      v68 = v21;
      sub_2B57670((__int64)v72, v51, &v67, &v68, (unsigned int *)&v66);
    }
    while ( v65 != v49 );
    v27 = v69;
  }
  LOBYTE(v32) = 0;
LABEL_34:
  if ( v27 != (__int64 *)v71 )
  {
    v64 = v32;
    _libc_free((unsigned __int64)v27);
    LOBYTE(v32) = v64;
  }
  return v32;
}
