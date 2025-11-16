// Function: sub_B20570
// Address: 0xb20570
//
__int64 __fastcall sub_B20570(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v7; // rcx
  __int64 result; // rax
  unsigned int v9; // r13d
  __int64 v10; // r14
  __int64 *v11; // rbx
  __int64 v12; // rcx
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rsi
  int v23; // edx
  char v24; // dl
  __int64 *v25; // rsi
  int v26; // ecx
  unsigned int v27; // eax
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 *v31; // r13
  __int64 *v32; // rbx
  __int64 v33; // rsi
  unsigned int v34; // eax
  __int64 v35; // r12
  __int64 *v36; // r15
  __int64 *v37; // r14
  unsigned int v38; // r13d
  __int64 v39; // rcx
  unsigned int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r14
  __int64 v46; // rdx
  _QWORD *v47; // rcx
  int v48; // edx
  __int64 v49; // rcx
  __int64 v50; // r14
  int v51; // r9d
  __int64 *v52; // [rsp+8h] [rbp-A8h]
  __int64 *v53; // [rsp+8h] [rbp-A8h]
  int v54; // [rsp+8h] [rbp-A8h]
  __int64 *v55; // [rsp+10h] [rbp-A0h]
  _BYTE *v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+18h] [rbp-98h]
  _BYTE *v59; // [rsp+30h] [rbp-80h] BYREF
  __int64 v60; // [rsp+38h] [rbp-78h]
  _BYTE v61[112]; // [rsp+40h] [rbp-70h] BYREF

  v4 = a1;
  if ( a3 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
    result = v7;
  }
  else
  {
    v7 = 0;
    result = 0;
  }
  v9 = *(_DWORD *)(a1 + 32);
  if ( v9 <= (unsigned int)result )
    return result;
  v10 = *(_QWORD *)(a1 + 24);
  v11 = *(__int64 **)(v10 + 8 * v7);
  if ( !v11 )
    return result;
  if ( a4 )
  {
    v12 = (unsigned int)(*(_DWORD *)(a4 + 44) + 1);
    result = v12;
  }
  else
  {
    v12 = 0;
    result = 0;
  }
  if ( v9 <= (unsigned int)result )
    return result;
  v13 = *(__int64 **)(v10 + 8 * v12);
  if ( !v13 )
    return result;
  v14 = sub_B192F0(a1, a3, a4);
  if ( v14 )
  {
    v15 = (unsigned int)(*(_DWORD *)(v14 + 44) + 1);
    result = v15;
  }
  else
  {
    v15 = 0;
    result = 0;
  }
  if ( v9 > (unsigned int)result && v13 == *(__int64 **)(v10 + 8 * v15) )
    return result;
  *(_BYTE *)(a1 + 112) = 0;
  if ( v11 != (__int64 *)v13[1] )
    return sub_B20080(v4, a2, *v11, *v13);
  v57 = *v13;
  if ( !a2 )
  {
    v42 = sub_B18F80(v57);
    v44 = v43;
    v59 = v61;
    v60 = 0x800000000LL;
    if ( v42 == v43 )
    {
      v48 = 0;
    }
    else
    {
      v45 = v42;
      v46 = 0;
      do
      {
        do
          v45 = *(_QWORD *)(v45 + 8);
        while ( v45 && (unsigned __int8)(**(_BYTE **)(v45 + 24) - 30) > 0xAu );
        ++v46;
      }
      while ( v44 != v45 );
      v47 = v61;
      if ( v46 > 8 )
      {
        v54 = v46;
        sub_C8D5F0(&v59, v61, v46, 8);
        LODWORD(v46) = v54;
        v47 = &v59[8 * (unsigned int)v60];
      }
      do
      {
        if ( v47 )
          *v47 = *(_QWORD *)(*(_QWORD *)(v42 + 24) + 40LL);
        do
          v42 = *(_QWORD *)(v42 + 8);
        while ( v42 && (unsigned __int8)(**(_BYTE **)(v42 + 24) - 30) > 0xAu );
        ++v47;
      }
      while ( v42 != v45 );
      v48 = v60 + v46;
    }
    v25 = 0;
    LODWORD(v60) = v48;
    sub_B1C8F0((__int64)&v59);
    goto LABEL_44;
  }
  v16 = *(_QWORD *)(*v13 + 16);
  v17 = *(_QWORD *)(a2 + 8);
  if ( v16 )
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v16 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v18 - 30) <= 0xAu )
        break;
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        goto LABEL_69;
    }
    v19 = 0;
    v59 = v61;
    v60 = 0x800000000LL;
    v20 = v16;
    while ( 1 )
    {
      v20 = *(_QWORD *)(v20 + 8);
      if ( !v20 )
        break;
      while ( (unsigned __int8)(**(_BYTE **)(v20 + 24) - 30) <= 0xAu )
      {
        v20 = *(_QWORD *)(v20 + 8);
        ++v19;
        if ( !v20 )
          goto LABEL_25;
      }
    }
LABEL_25:
    v21 = v19 + 1;
    if ( v19 + 1 > 8 )
    {
      sub_C8D5F0(&v59, v61, v21, 8);
      LODWORD(v21) = v19 + 1;
      v22 = &v59[8 * (unsigned int)v60];
      v18 = *(_QWORD *)(v16 + 24);
    }
    else
    {
      v22 = v61;
    }
LABEL_29:
    if ( v22 )
      *v22 = *(_QWORD *)(v18 + 40);
    while ( 1 )
    {
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        break;
      v18 = *(_QWORD *)(v16 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v18 - 30) <= 0xAu )
      {
        ++v22;
        goto LABEL_29;
      }
    }
    v23 = v60 + v21;
  }
  else
  {
LABEL_69:
    HIDWORD(v60) = 8;
    v23 = 0;
    v59 = v61;
  }
  LODWORD(v60) = v23;
  sub_B1C8F0((__int64)&v59);
  v24 = *(_BYTE *)(v17 + 312) & 1;
  if ( v24 )
  {
    v25 = (__int64 *)(v17 + 320);
    v26 = 3;
  }
  else
  {
    v49 = *(unsigned int *)(v17 + 328);
    v25 = *(__int64 **)(v17 + 320);
    if ( !(_DWORD)v49 )
      goto LABEL_80;
    v26 = v49 - 1;
  }
  v27 = v26 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
  v55 = &v25[9 * v27];
  v28 = *v55;
  if ( v57 != *v55 )
  {
    v51 = 1;
    while ( v28 != -4096 )
    {
      v27 = v26 & (v51 + v27);
      v55 = &v25[9 * v27];
      v28 = *v55;
      if ( v57 == *v55 )
        goto LABEL_36;
      ++v51;
    }
    if ( v24 )
    {
      v50 = 36;
      goto LABEL_81;
    }
    v49 = *(unsigned int *)(v17 + 328);
LABEL_80:
    v50 = 9 * v49;
LABEL_81:
    v55 = &v25[v50];
  }
LABEL_36:
  v29 = 36;
  if ( !v24 )
    v29 = 9LL * *(unsigned int *)(v17 + 328);
  if ( v55 != &v25[v29] )
  {
    v30 = (__int64 *)v55[1];
    v31 = &v30[*((unsigned int *)v55 + 4)];
    if ( v30 != v31 )
    {
      v52 = v11;
      v32 = (__int64 *)v55[1];
      do
      {
        v33 = *v32++;
        sub_B1CA60((__int64)&v59, v33);
      }
      while ( v31 != v32 );
      v11 = v52;
    }
    v25 = v55 + 5;
    sub_B1CB00((__int64)&v59, (__int64)(v55 + 5));
  }
LABEL_44:
  v56 = v59;
  if ( v59 != &v59[8 * (unsigned int)v60] )
  {
    v34 = *(_DWORD *)(v4 + 32);
    v53 = v13;
    v35 = v4;
    v36 = (__int64 *)v59;
    v37 = (__int64 *)&v59[8 * (unsigned int)v60];
    v38 = v34;
    while ( 1 )
    {
      v41 = *v36;
      if ( *v36 )
      {
        v39 = (unsigned int)(*(_DWORD *)(v41 + 44) + 1);
        v40 = *(_DWORD *)(v41 + 44) + 1;
      }
      else
      {
        v39 = 0;
        v40 = 0;
      }
      if ( v40 < v38 )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v35 + 24) + 8 * v39) )
        {
          v25 = (__int64 *)v57;
          if ( v57 != sub_B192F0(v35, v57, v41) )
            break;
        }
      }
      if ( v37 == ++v36 )
      {
        v4 = v35;
        v13 = v53;
        goto LABEL_71;
      }
    }
    v4 = v35;
    v13 = v53;
    if ( v56 != v61 )
      _libc_free(v56, v57);
    return sub_B20080(v4, a2, *v11, *v13);
  }
LABEL_71:
  if ( v56 != v61 )
    _libc_free(v56, v25);
  return sub_B1F4E0(v4, a2, (__int64)v13);
}
