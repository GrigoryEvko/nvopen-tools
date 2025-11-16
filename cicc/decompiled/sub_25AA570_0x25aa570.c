// Function: sub_25AA570
// Address: 0x25aa570
//
void __fastcall sub_25AA570(unsigned __int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // rax
  const void *v8; // rdi
  char v9; // dl
  __int64 v10; // rax
  _QWORD *v11; // rcx
  _QWORD *v12; // r15
  _BYTE *v13; // r13
  _QWORD *v14; // rdx
  __int64 v15; // rax
  size_t v16; // rdx
  _BYTE *v17; // rax
  _BYTE *v18; // rsi
  int v19; // r12d
  unsigned __int64 v20; // r9
  _BYTE *v21; // r14
  __int64 v22; // rax
  void *v23; // r10
  void *v24; // rax
  unsigned __int64 v25; // r12
  char *v26; // rdi
  const void *v27; // rsi
  __int64 v28; // r12
  _QWORD *v29; // r12
  __int64 v30; // rsi
  _BYTE *v31; // rbx
  unsigned __int64 v32; // rdi
  int v33; // eax
  _QWORD *v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  _QWORD *v37; // r15
  __int64 v38; // r14
  __int64 v39; // rax
  _QWORD *v40; // rdx
  unsigned __int64 v41; // r13
  _QWORD *v42; // rax
  char v43; // di
  unsigned __int64 v44; // rsi
  _QWORD *v45; // rcx
  char v46; // r15
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 *v51; // rax
  __int64 *v52; // rdx
  __int64 i; // r13
  __int64 v54; // rax
  _QWORD *v55; // [rsp+10h] [rbp-310h]
  size_t nc; // [rsp+20h] [rbp-300h]
  size_t n; // [rsp+20h] [rbp-300h]
  size_t nd; // [rsp+20h] [rbp-300h]
  char na; // [rsp+20h] [rbp-300h]
  _QWORD *ne; // [rsp+20h] [rbp-300h]
  size_t nb; // [rsp+20h] [rbp-300h]
  _QWORD *v63; // [rsp+28h] [rbp-2F8h]
  __int64 v64; // [rsp+28h] [rbp-2F8h]
  _QWORD *v65; // [rsp+30h] [rbp-2F0h] BYREF
  const void *v66; // [rsp+38h] [rbp-2E8h]
  __int64 v67; // [rsp+40h] [rbp-2E0h]
  _QWORD v68[3]; // [rsp+48h] [rbp-2D8h] BYREF
  __int64 v69; // [rsp+60h] [rbp-2C0h] BYREF
  __int64 v70; // [rsp+68h] [rbp-2B8h]
  _QWORD *v71; // [rsp+70h] [rbp-2B0h] BYREF
  unsigned int v72; // [rsp+78h] [rbp-2A8h]
  _BYTE v73[48]; // [rsp+2F0h] [rbp-30h] BYREF

  v6 = &v71;
  v69 = 0;
  v70 = 1;
  do
  {
    *(_QWORD *)v6 = -2;
    v6 += 40;
  }
  while ( v6 != v73 );
  v8 = *(const void **)a1;
  sub_25AA1B0((size_t)v8, (__int64)a2, (__int64)&v69, (size_t *)a1, a5, a6);
  if ( (unsigned int)v70 >> 1 )
  {
    v9 = v70 & 1;
    if ( (v70 & 1) != 0 )
    {
      v13 = v73;
      v12 = &v71;
    }
    else
    {
      v10 = v72;
      v11 = v71;
      v12 = v71;
      v13 = &v71[5 * v72];
      if ( v71 == (_QWORD *)v13 )
      {
LABEL_11:
        v14 = v11;
        v15 = 5 * v10;
        goto LABEL_12;
      }
    }
    do
    {
      if ( *v12 != -2 && *v12 != -16 )
        break;
      v12 += 5;
    }
    while ( v12 != (_QWORD *)v13 );
  }
  else
  {
    v9 = v70 & 1;
    if ( (v70 & 1) != 0 )
    {
      v35 = 80;
      v34 = &v71;
    }
    else
    {
      v34 = v71;
      v35 = 5LL * v72;
    }
    v12 = &v34[v35];
    v13 = &v34[v35];
  }
  if ( !v9 )
  {
    v11 = v71;
    v10 = v72;
    goto LABEL_11;
  }
  v15 = 80;
  v14 = &v71;
LABEL_12:
  v63 = &v14[v15];
  while ( v63 != v12 )
  {
    v16 = *(_QWORD *)a1;
    v17 = *(_BYTE **)(*(_QWORD *)a1 + 88LL);
    v18 = *(_BYTE **)(*(_QWORD *)a1 + 80LL);
    v19 = *(_DWORD *)(*(_QWORD *)a1 + 72LL);
    v20 = v17 - v18;
    v21 = (_BYTE *)(v17 - v18);
    if ( v17 == v18 )
    {
      v23 = 0;
    }
    else
    {
      if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_105;
      v8 = (const void *)(*(_QWORD *)(*(_QWORD *)a1 + 88LL) - (_QWORD)v18);
      nc = *(_QWORD *)a1;
      v22 = sub_22077B0(v20);
      v16 = nc;
      v23 = (void *)v22;
      v17 = *(_BYTE **)(nc + 88);
      v18 = *(_BYTE **)(nc + 80);
      v20 = v17 - v18;
    }
    if ( v18 == v17 )
    {
      if ( v19 != *((_DWORD *)v12 + 2) || (v8 = (const void *)v12[2], v12[3] - (_QWORD)v8 != v20) )
      {
        if ( v23 )
        {
LABEL_18:
          v18 = v21;
          v8 = v23;
          j_j___libc_free_0((unsigned __int64)v23);
        }
        LODWORD(v65) = *((_DWORD *)v12 + 2);
        v25 = v12[3] - v12[2];
        v66 = 0;
        v67 = 0;
        v68[0] = 0;
        if ( v25 )
        {
          if ( v25 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_105:
            sub_4261EA(v8, v18, v16);
          v26 = (char *)sub_22077B0(v25);
        }
        else
        {
          v26 = 0;
        }
        v66 = v26;
        v68[0] = &v26[v25];
        v67 = (__int64)v26;
        v27 = (const void *)v12[2];
        v28 = v12[3] - (_QWORD)v27;
        if ( (const void *)v12[3] != v27 )
          v26 = (char *)memmove(v26, v27, v12[3] - (_QWORD)v27);
        v67 = (__int64)&v26[v28];
        sub_25A6B00(a1, *v12, (__int64)&v65);
        v8 = v66;
        if ( v66 )
          j_j___libc_free_0((unsigned __int64)v66);
        goto LABEL_26;
      }
      if ( v20 )
      {
LABEL_48:
        nd = (size_t)v23;
        v33 = memcmp(v8, v23, v20);
        v23 = (void *)nd;
        if ( v33 )
          goto LABEL_18;
LABEL_49:
        v8 = v23;
        j_j___libc_free_0((unsigned __int64)v23);
        goto LABEL_26;
      }
    }
    else
    {
      n = v20;
      v24 = memmove(v23, v18, v20);
      v20 = n;
      v23 = v24;
      if ( v19 != *((_DWORD *)v12 + 2) )
        goto LABEL_18;
      v8 = (const void *)v12[2];
      if ( n != v12[3] - (_QWORD)v8 )
        goto LABEL_18;
      if ( n )
        goto LABEL_48;
    }
    if ( v23 )
      goto LABEL_49;
    do
LABEL_26:
      v12 += 5;
    while ( v12 != (_QWORD *)v13 && (*v12 == -2 || *v12 == -16) );
  }
  if ( (unsigned int)*a2 - 30 <= 0xA )
  {
    v65 = v68;
    v66 = 0;
    v67 = 16;
    sub_25A8650(a1, (__int64)a2, &v65, 1);
    v36 = *((_QWORD *)a2 + 5);
    v37 = v65;
    if ( !(_DWORD)v66 )
    {
LABEL_92:
      if ( v37 != v68 )
        _libc_free((unsigned __int64)v37);
      goto LABEL_31;
    }
    v38 = 0;
    v64 = (unsigned int)v66;
    v55 = (_QWORD *)(a1 + 1264);
LABEL_65:
    while ( 2 )
    {
      if ( !*((_BYTE *)v37 + v38) )
        goto LABEL_64;
      na = *((_BYTE *)v37 + v38);
      v39 = sub_B46EC0((__int64)a2, v38);
      v40 = *(_QWORD **)(a1 + 1272);
      v41 = v39;
      if ( v40 )
      {
        while ( 1 )
        {
          v44 = v40[4];
          if ( v36 < v44 || v36 == v44 && v41 < v40[5] )
          {
            v42 = (_QWORD *)v40[2];
            v43 = na;
            if ( !v42 )
            {
LABEL_74:
              if ( v43 )
              {
                if ( *(_QWORD **)(a1 + 1280) == v40 )
                  goto LABEL_80;
                goto LABEL_95;
              }
              v45 = v40;
              if ( v36 <= v44 )
              {
LABEL_76:
                if ( v36 == v44 && v41 > v40[5] )
                {
                  v40 = v45;
                  goto LABEL_79;
                }
LABEL_64:
                if ( v64 == ++v38 )
                  goto LABEL_92;
                goto LABEL_65;
              }
LABEL_80:
              v46 = 1;
              if ( v55 != v40 && v36 >= v40[4] )
              {
                v46 = 0;
                if ( v36 == v40[4] )
                  v46 = v41 < v40[5];
              }
              goto LABEL_81;
            }
          }
          else
          {
            v42 = (_QWORD *)v40[3];
            v43 = 0;
            if ( !v42 )
              goto LABEL_74;
          }
          v40 = v42;
        }
      }
      v40 = (_QWORD *)(a1 + 1264);
      if ( v55 != *(_QWORD **)(a1 + 1280) )
      {
LABEL_95:
        nb = (size_t)v40;
        v54 = sub_220EF80((__int64)v40);
        v40 = (_QWORD *)nb;
        v44 = *(_QWORD *)(v54 + 32);
        if ( v44 >= v36 )
        {
          v45 = (_QWORD *)nb;
          v40 = (_QWORD *)v54;
          goto LABEL_76;
        }
LABEL_79:
        if ( !v40 )
          goto LABEL_64;
        goto LABEL_80;
      }
      v40 = (_QWORD *)(a1 + 1264);
      v46 = 1;
LABEL_81:
      ne = v40;
      v47 = sub_22077B0(0x30u);
      *(_QWORD *)(v47 + 32) = v36;
      *(_QWORD *)(v47 + 40) = v41;
      sub_220F040(v46, v47, ne, v55);
      ++*(_QWORD *)(a1 + 1296);
      if ( *(_BYTE *)(a1 + 68) )
      {
        v51 = *(__int64 **)(a1 + 48);
        v52 = &v51[*(unsigned int *)(a1 + 60)];
        if ( v51 == v52 )
          goto LABEL_99;
        while ( v41 != *v51 )
        {
          if ( v52 == ++v51 )
            goto LABEL_99;
        }
      }
      else if ( !sub_C8CA60(a1 + 40, v41) )
      {
LABEL_99:
        sub_25A5A10(a1, v41, v52, v48, v49, v50);
        v37 = v65;
        goto LABEL_91;
      }
      for ( i = *(_QWORD *)(v41 + 56); ; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        if ( *(_BYTE *)(i - 24) != 84 )
          break;
        sub_25A8730((size_t *)a1, i - 24);
      }
      v37 = v65;
LABEL_91:
      if ( v64 == ++v38 )
        goto LABEL_92;
      continue;
    }
  }
LABEL_31:
  if ( (v70 & 1) != 0 )
  {
    v31 = v73;
    v29 = &v71;
  }
  else
  {
    v29 = v71;
    v30 = 5LL * v72;
    if ( !v72 )
      goto LABEL_57;
    v31 = &v71[v30];
    if ( v71 == &v71[v30] )
      goto LABEL_57;
  }
  do
  {
    if ( *v29 != -2 && *v29 != -16 )
    {
      v32 = v29[2];
      if ( v32 )
        j_j___libc_free_0(v32);
    }
    v29 += 5;
  }
  while ( v29 != (_QWORD *)v31 );
  if ( (v70 & 1) == 0 )
  {
    v29 = v71;
    v30 = 5LL * v72;
LABEL_57:
    sub_C7D6A0((__int64)v29, v30 * 8, 8);
  }
}
