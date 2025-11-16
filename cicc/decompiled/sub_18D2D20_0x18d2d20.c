// Function: sub_18D2D20
// Address: 0x18d2d20
//
void __fastcall sub_18D2D20(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v4; // eax
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r9
  int v8; // r13d
  __int64 *v9; // rcx
  unsigned int v10; // r8d
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r15
  unsigned int v14; // esi
  __int64 v15; // r15
  int v16; // eax
  int v17; // esi
  __int64 v18; // rdx
  unsigned int v19; // eax
  int v20; // edi
  __int64 v21; // r8
  int v22; // r10d
  __int64 *v23; // r9
  _QWORD *v24; // r14
  _QWORD *v25; // r12
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r10
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // r13
  __int64 v33; // r15
  __int64 v34; // r15
  int v35; // edx
  int v36; // r11d
  _QWORD *v37; // r14
  _QWORD *v38; // r13
  _QWORD *v39; // rbx
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  int v42; // eax
  int v43; // eax
  __int64 v44; // r8
  int v45; // r10d
  unsigned int v46; // edx
  __int64 v47; // rsi
  _QWORD *v48; // r13
  _QWORD *v49; // rbx
  unsigned __int64 v50; // rdi
  unsigned __int64 v51; // rdi
  unsigned int v52; // [rsp-DCh] [rbp-DCh]
  __int64 v53; // [rsp-D8h] [rbp-D8h]
  __int64 v55; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v56; // [rsp-C0h] [rbp-C0h]
  __int64 v57; // [rsp-B8h] [rbp-B8h]
  __int64 v58; // [rsp-B0h] [rbp-B0h] BYREF
  _BYTE *v59; // [rsp-A8h] [rbp-A8h]
  _BYTE *v60; // [rsp-A0h] [rbp-A0h]
  __int64 v61; // [rsp-98h] [rbp-98h]
  int v62; // [rsp-90h] [rbp-90h]
  _BYTE v63[16]; // [rsp-88h] [rbp-88h] BYREF
  __int64 v64; // [rsp-78h] [rbp-78h] BYREF
  _BYTE *v65; // [rsp-70h] [rbp-70h]
  _BYTE *v66; // [rsp-68h] [rbp-68h]
  __int64 v67; // [rsp-60h] [rbp-60h]
  int v68; // [rsp-58h] [rbp-58h]
  _BYTE v69[16]; // [rsp-50h] [rbp-50h] BYREF
  char v70; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)a1 == -1 )
    return;
  v2 = a2;
  v4 = *(_DWORD *)a2 + *(_DWORD *)a1;
  *(_DWORD *)a1 = v4;
  if ( v4 == -1 )
  {
    sub_18CEB70(a1 + 8);
    v37 = *(_QWORD **)(a1 + 40);
    v48 = *(_QWORD **)(a1 + 48);
    if ( v37 == v48 )
      return;
    v49 = *(_QWORD **)(a1 + 40);
    do
    {
      v50 = v49[13];
      if ( v50 != v49[12] )
        _libc_free(v50);
      v51 = v49[6];
      if ( v51 != v49[5] )
        _libc_free(v51);
      v49 += 19;
    }
    while ( v48 != v49 );
LABEL_64:
    *(_QWORD *)(a1 + 48) = v37;
    return;
  }
  if ( v4 < *(_DWORD *)a2 )
  {
    *(_DWORD *)a1 = -1;
    sub_18CEB70(a1 + 8);
    v37 = *(_QWORD **)(a1 + 40);
    v38 = *(_QWORD **)(a1 + 48);
    if ( v37 == v38 )
      return;
    v39 = *(_QWORD **)(a1 + 40);
    do
    {
      v40 = v39[13];
      if ( v40 != v39[12] )
        _libc_free(v40);
      v41 = v39[6];
      if ( v41 != v39[5] )
        _libc_free(v41);
      v39 += 19;
    }
    while ( v38 != v39 );
    goto LABEL_64;
  }
  v5 = *(_QWORD *)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 40);
  v53 = a1 + 8;
  if ( v6 == v5 )
    goto LABEL_23;
  do
  {
    v14 = *(_DWORD *)(a1 + 32);
    v15 = *(_QWORD *)v6;
    if ( v14 )
    {
      v7 = *(_QWORD *)(a1 + 16);
      v8 = 1;
      v9 = 0;
      v10 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v11 = (__int64 *)(v7 + 16LL * v10);
      v12 = *v11;
      if ( v15 == *v11 )
      {
LABEL_7:
        v13 = *(_QWORD *)(a1 + 40) + 152 * v11[1] + 8;
        LOWORD(v55) = *(_WORD *)(v6 + 8);
        BYTE2(v55) = *(_BYTE *)(v6 + 10);
        LOWORD(v56) = *(_WORD *)(v6 + 16);
        v57 = *(_QWORD *)(v6 + 24);
        sub_16CCCB0(&v58, (__int64)v63, v6 + 32);
        sub_16CCCB0(&v64, (__int64)v69, v6 + 88);
        v70 = *(_BYTE *)(v6 + 144);
        goto LABEL_8;
      }
      while ( v12 != -8 )
      {
        if ( !v9 && v12 == -16 )
          v9 = v11;
        v10 = (v14 - 1) & (v8 + v10);
        v11 = (__int64 *)(v7 + 16LL * v10);
        v12 = *v11;
        if ( v15 == *v11 )
          goto LABEL_7;
        ++v8;
      }
      if ( !v9 )
        v9 = v11;
      v31 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v20 = v31 + 1;
      if ( 4 * (v31 + 1) < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(a1 + 28) - v20 > v14 >> 3 )
          goto LABEL_44;
        v52 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
        sub_18D2390(v53, v14);
        v42 = *(_DWORD *)(a1 + 32);
        if ( !v42 )
        {
LABEL_86:
          ++*(_DWORD *)(a1 + 24);
          BUG();
        }
        v43 = v42 - 1;
        v44 = *(_QWORD *)(a1 + 16);
        v23 = 0;
        v45 = 1;
        v20 = *(_DWORD *)(a1 + 24) + 1;
        v46 = v43 & v52;
        v9 = (__int64 *)(v44 + 16LL * (v43 & v52));
        v47 = *v9;
        if ( v15 == *v9 )
          goto LABEL_44;
        while ( v47 != -8 )
        {
          if ( v47 == -16 && !v23 )
            v23 = v9;
          v46 = v43 & (v46 + v45);
          v9 = (__int64 *)(v44 + 16LL * v46);
          v47 = *v9;
          if ( v15 == *v9 )
            goto LABEL_44;
          ++v45;
        }
        goto LABEL_19;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
    }
    sub_18D2390(v53, 2 * v14);
    v16 = *(_DWORD *)(a1 + 32);
    if ( !v16 )
      goto LABEL_86;
    v17 = v16 - 1;
    v18 = *(_QWORD *)(a1 + 16);
    v19 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v20 = *(_DWORD *)(a1 + 24) + 1;
    v9 = (__int64 *)(v18 + 16LL * v19);
    v21 = *v9;
    if ( v15 == *v9 )
      goto LABEL_44;
    v22 = 1;
    v23 = 0;
    while ( v21 != -8 )
    {
      if ( !v23 && v21 == -16 )
        v23 = v9;
      v19 = v17 & (v22 + v19);
      v9 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v9;
      if ( v15 == *v9 )
        goto LABEL_44;
      ++v22;
    }
LABEL_19:
    if ( v23 )
      v9 = v23;
LABEL_44:
    *(_DWORD *)(a1 + 24) = v20;
    if ( *v9 != -8 )
      --*(_DWORD *)(a1 + 28);
    *v9 = v15;
    v9[1] = 0;
    v32 = *(_QWORD *)(a1 + 48) - *(_QWORD *)(a1 + 40);
    v9[1] = 0x86BCA1AF286BCA1BLL * (v32 >> 3);
    v33 = *(_QWORD *)(a1 + 48);
    if ( v33 == *(_QWORD *)(a1 + 56) )
    {
      sub_18D1050((__int64 *)(a1 + 40), *(char **)(a1 + 48), (__int64 *)v6);
    }
    else
    {
      if ( v33 )
      {
        *(_QWORD *)v33 = *(_QWORD *)v6;
        *(_BYTE *)(v33 + 8) = *(_BYTE *)(v6 + 8);
        *(_BYTE *)(v33 + 9) = *(_BYTE *)(v6 + 9);
        *(_BYTE *)(v33 + 10) = *(_BYTE *)(v6 + 10);
        *(_BYTE *)(v33 + 16) = *(_BYTE *)(v6 + 16);
        *(_BYTE *)(v33 + 17) = *(_BYTE *)(v6 + 17);
        *(_QWORD *)(v33 + 24) = *(_QWORD *)(v6 + 24);
        sub_16CCCB0((_QWORD *)(v33 + 32), v33 + 72, v6 + 32);
        sub_16CCCB0((_QWORD *)(v33 + 88), v33 + 128, v6 + 88);
        *(_BYTE *)(v33 + 144) = *(_BYTE *)(v6 + 144);
        v33 = *(_QWORD *)(a1 + 48);
      }
      *(_QWORD *)(a1 + 48) = v33 + 152;
    }
    v34 = *(_QWORD *)(a1 + 40);
    v55 = 0;
    v59 = v63;
    v60 = v63;
    v56 = 0;
    v13 = v32 + v34 + 8;
    v57 = 0;
    v58 = 0;
    v61 = 2;
    v62 = 0;
    v64 = 0;
    v65 = v69;
    v66 = v69;
    v67 = 2;
    v68 = 0;
    v70 = 0;
LABEL_8:
    sub_18DBA00(v13, &v55, 1);
    if ( v66 != v65 )
      _libc_free((unsigned __int64)v66);
    if ( v60 != v59 )
      _libc_free((unsigned __int64)v60);
    v6 += 152;
  }
  while ( v5 != v6 );
  v2 = a2;
LABEL_23:
  v24 = *(_QWORD **)(a1 + 48);
  v25 = *(_QWORD **)(a1 + 40);
  if ( v24 != v25 )
  {
    while ( 1 )
    {
      v30 = *(unsigned int *)(v2 + 32);
      if ( !(_DWORD)v30 )
        goto LABEL_30;
      v26 = *(_QWORD *)(v2 + 16);
      v27 = (v30 - 1) & (((unsigned int)*v25 >> 9) ^ ((unsigned int)*v25 >> 4));
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( *v25 != *v28 )
        break;
LABEL_26:
      if ( v28 == (__int64 *)(v26 + 16 * v30) || *(_QWORD *)(v2 + 48) == *(_QWORD *)(v2 + 40) + 152 * v28[1] )
        goto LABEL_30;
LABEL_28:
      v25 += 19;
      if ( v24 == v25 )
        return;
    }
    v35 = 1;
    while ( v29 != -8 )
    {
      v36 = v35 + 1;
      v27 = (v30 - 1) & (v35 + v27);
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( *v25 == *v28 )
        goto LABEL_26;
      v35 = v36;
    }
LABEL_30:
    v70 = 0;
    v59 = v63;
    v60 = v63;
    v55 = 0;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v61 = 2;
    v62 = 0;
    v64 = 0;
    v65 = v69;
    v66 = v69;
    v67 = 2;
    v68 = 0;
    sub_18DBA00(v25 + 1, &v55, 1);
    if ( v66 != v65 )
      _libc_free((unsigned __int64)v66);
    if ( v60 != v59 )
      _libc_free((unsigned __int64)v60);
    goto LABEL_28;
  }
}
