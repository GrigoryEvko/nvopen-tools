// Function: sub_29E73A0
// Address: 0x29e73a0
//
_BYTE *__fastcall sub_29E73A0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  _BYTE *result; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int8 *v13; // rsi
  unsigned __int8 *v14; // r12
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 *v18; // rax
  _BYTE *v19; // rax
  _QWORD *v20; // r11
  __int64 j; // rax
  __int64 v22; // r8
  __int64 v23; // rdi
  int v24; // r10d
  __int64 v25; // rcx
  __int64 *v26; // rdx
  __int64 v27; // r9
  _QWORD *v28; // rax
  __int64 v29; // rax
  __int64 *v30; // r9
  __int64 *v31; // r14
  __int64 *v32; // r12
  __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 i; // r13
  char *v37; // r15
  char v38; // al
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned int v41; // r10d
  __int64 v42; // r14
  char *v43; // r12
  char v44; // dl
  int v45; // esi
  int v46; // ecx
  __int64 *v47; // rax
  unsigned __int8 *v48; // r14
  unsigned int v49; // r15d
  int v50; // r12d
  int v51; // r13d
  unsigned __int8 **v52; // r12
  int v53; // r11d
  int v54; // edi
  int v55; // eax
  _BYTE *v56; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v57; // [rsp+18h] [rbp-A8h] BYREF
  unsigned __int8 *v58; // [rsp+28h] [rbp-98h] BYREF
  unsigned __int8 *v59; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v60; // [rsp+38h] [rbp-88h] BYREF
  _QWORD *v61; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v62; // [rsp+48h] [rbp-78h]
  unsigned int v63; // [rsp+4Ch] [rbp-74h]
  _QWORD v64[14]; // [rsp+50h] [rbp-70h] BYREF

  v57 = a1;
  if ( *a1 == 81 )
  {
    a1 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
    v57 = a1;
  }
  v7 = *(unsigned int *)(a2 + 24);
  v8 = *(_QWORD *)(a2 + 8);
  if ( (_DWORD)v7 )
  {
    a5 = (unsigned int)(v7 - 1);
    a4 = (unsigned int)a5 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    a3 = v8 + 16 * a4;
    a6 = *(_QWORD *)a3;
    if ( a1 == *(unsigned __int8 **)a3 )
    {
LABEL_5:
      if ( a3 != v8 + 16 * v7 )
        return *(_BYTE **)(a3 + 8);
    }
    else
    {
      a3 = 1;
      while ( a6 != -4096 )
      {
        v41 = a3 + 1;
        a4 = (unsigned int)a5 & ((_DWORD)a3 + (_DWORD)a4);
        a3 = v8 + 16LL * (unsigned int)a4;
        a6 = *(_QWORD *)a3;
        if ( *(unsigned __int8 **)a3 == a1 )
          goto LABEL_5;
        a3 = v41;
      }
    }
  }
  result = sub_29E69F0(a1, a2, a3, a4, a5, a6);
  if ( result )
    return result;
  *sub_29E7150(a2, (__int64 *)&v57) = 0;
  v13 = v57;
  v58 = v57;
  if ( (unsigned __int8)(*v57 - 80) <= 1u )
    v14 = (unsigned __int8 *)*((_QWORD *)v57 - 4);
  else
    v14 = (unsigned __int8 *)**((_QWORD **)v57 - 1);
  v15 = *v14;
  v56 = 0;
  if ( *v14 > 0x1Cu )
  {
    while ( 1 )
    {
      if ( v15 == 81 )
        goto LABEL_21;
      v16 = *(unsigned int *)(a2 + 24);
      if ( !(_DWORD)v16 )
        goto LABEL_86;
      v17 = *(_QWORD *)(a2 + 8);
      v10 = ((_DWORD)v16 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v18 = (__int64 *)(v17 + 16 * v10);
      v11 = *v18;
      if ( (unsigned __int8 *)*v18 != v14 )
        break;
LABEL_17:
      v16 = v17 + 16 * v16;
      if ( v18 == (__int64 *)v16 )
        goto LABEL_86;
      v19 = (_BYTE *)v18[1];
LABEL_19:
      if ( v19 )
      {
        v56 = v19;
        v13 = v58;
        goto LABEL_25;
      }
      v58 = v14;
      *sub_29E7150(a2, (__int64 *)&v58) = 0;
LABEL_21:
      if ( (unsigned __int8)(*v14 - 80) > 1u )
        v14 = (unsigned __int8 *)**((_QWORD **)v14 - 1);
      else
        v14 = (unsigned __int8 *)*((_QWORD *)v14 - 4);
      v15 = *v14;
      if ( *v14 <= 0x1Cu )
      {
        v56 = 0;
        v13 = v58;
        goto LABEL_25;
      }
    }
    v55 = 1;
    while ( v11 != -4096 )
    {
      v12 = (unsigned int)(v55 + 1);
      v10 = ((_DWORD)v16 - 1) & (unsigned int)(v55 + v10);
      v18 = (__int64 *)(v17 + 16LL * (unsigned int)v10);
      v11 = *v18;
      if ( (unsigned __int8 *)*v18 == v14 )
        goto LABEL_17;
      v55 = v12;
    }
LABEL_86:
    v19 = sub_29E69F0(v14, a2, v16, v10, v11, v12);
    goto LABEL_19;
  }
LABEL_25:
  v63 = 8;
  v20 = v64;
  v61 = v64;
  LODWORD(j) = 1;
  v64[0] = v13;
  while ( 1 )
  {
    v22 = *(unsigned int *)(a2 + 24);
    LODWORD(j) = j - 1;
    v59 = v13;
    v62 = j;
    v23 = *(_QWORD *)(a2 + 8);
    if ( !(_DWORD)v22 )
    {
      ++*(_QWORD *)a2;
      v60 = 0;
      goto LABEL_66;
    }
    v24 = v22 - 1;
    LODWORD(v25) = (v22 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v26 = (__int64 *)(v23 + 16LL * (unsigned int)v25);
    v27 = *v26;
    if ( v13 != (unsigned __int8 *)*v26 )
    {
      v48 = (unsigned __int8 *)*v26;
      v49 = (v22 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v50 = 1;
      while ( v48 != (unsigned __int8 *)-4096LL )
      {
        v51 = v50 + 1;
        v49 = v24 & (v50 + v49);
        v52 = (unsigned __int8 **)(v23 + 16LL * v49);
        v48 = *v52;
        if ( v13 == *v52 )
        {
          if ( v52 == (unsigned __int8 **)(v23 + 16LL * (unsigned int)v22) )
            goto LABEL_79;
          v26 = (__int64 *)(v23 + 16LL * v49);
          goto LABEL_31;
        }
        v50 = v51;
      }
      goto LABEL_32;
    }
    if ( v26 != (__int64 *)(v23 + 16LL * (unsigned int)v22) )
    {
LABEL_31:
      if ( v26[1] )
        goto LABEL_26;
LABEL_32:
      v25 = v24 & (((unsigned int)v13 >> 4) ^ ((unsigned int)v13 >> 9));
      v26 = (__int64 *)(v23 + 16 * v25);
      v27 = *v26;
      if ( v13 == (unsigned __int8 *)*v26 )
        goto LABEL_33;
LABEL_79:
      v47 = 0;
      v53 = 1;
      while ( v27 != -4096 )
      {
        if ( v27 == -8192 && !v47 )
          v47 = v26;
        LODWORD(v25) = v24 & (v53 + v25);
        v26 = (__int64 *)(v23 + 16LL * (unsigned int)v25);
        v27 = *v26;
        if ( v13 == (unsigned __int8 *)*v26 )
          goto LABEL_33;
        ++v53;
      }
      v54 = *(_DWORD *)(a2 + 16);
      if ( !v47 )
        v47 = v26;
      ++*(_QWORD *)a2;
      v46 = v54 + 1;
      v60 = v47;
      if ( 4 * (v54 + 1) < (unsigned int)(3 * v22) )
      {
        if ( (int)v22 - *(_DWORD *)(a2 + 20) - v46 > (unsigned int)v22 >> 3 )
          goto LABEL_68;
        v45 = v22;
LABEL_67:
        sub_2685CE0(a2, v45);
        sub_2677F80(a2, (__int64 *)&v59, &v60);
        v13 = v59;
        v46 = *(_DWORD *)(a2 + 16) + 1;
        v47 = v60;
LABEL_68:
        *(_DWORD *)(a2 + 16) = v46;
        if ( *v47 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v47 = (__int64)v13;
        v28 = v47 + 1;
        *v28 = 0;
        goto LABEL_34;
      }
LABEL_66:
      v45 = 2 * v22;
      goto LABEL_67;
    }
LABEL_33:
    v28 = v26 + 1;
LABEL_34:
    *v28 = v56;
    if ( *v59 == 39 )
    {
      v29 = *((_QWORD *)v59 - 1);
      v30 = (__int64 *)(v29 + 64);
      v31 = (__int64 *)(v29 + 32LL * (*((_DWORD *)v59 + 1) & 0x7FFFFFF));
      if ( (v59[2] & 1) == 0 )
        v30 = (__int64 *)(v29 + 32);
      if ( v31 != v30 )
      {
        v32 = v30;
        do
        {
          v33 = sub_AA4FF0(*v32);
          if ( !v33 )
            BUG();
          for ( i = *(_QWORD *)(v33 - 8); i; i = *(_QWORD *)(i + 8) )
          {
            while ( 1 )
            {
              v37 = *(char **)(i + 24);
              v38 = *v37;
              if ( (unsigned __int8)*v37 > 0x1Cu && (v38 == 80 || v38 == 39) )
                break;
              i = *(_QWORD *)(i + 8);
              if ( !i )
                goto LABEL_49;
            }
            v39 = v62;
            v40 = v62 + 1LL;
            if ( v40 > v63 )
            {
              sub_C8D5F0((__int64)&v61, v64, v40, 8u, v34, v35);
              v39 = v62;
            }
            v61[v39] = v37;
            ++v62;
          }
LABEL_49:
          v32 += 4;
        }
        while ( v31 != v32 );
      }
      LODWORD(j) = v62;
    }
    else
    {
      v42 = *((_QWORD *)v59 + 2);
      for ( j = v62; v42; v42 = *(_QWORD *)(v42 + 8) )
      {
        v43 = *(char **)(v42 + 24);
        v44 = *v43;
        if ( (unsigned __int8)*v43 > 0x1Cu && (v44 == 80 || v44 == 39) )
        {
          if ( j + 1 > (unsigned __int64)v63 )
          {
            sub_C8D5F0((__int64)&v61, v64, j + 1, 8u, v22, v27);
            j = v62;
          }
          v61[j] = v43;
          j = ++v62;
        }
      }
    }
    v20 = v61;
LABEL_26:
    if ( !(_DWORD)j )
      break;
    v13 = (unsigned __int8 *)v20[(unsigned int)j - 1];
  }
  if ( v20 != v64 )
    _libc_free((unsigned __int64)v20);
  return v56;
}
