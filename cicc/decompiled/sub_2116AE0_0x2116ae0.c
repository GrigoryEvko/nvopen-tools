// Function: sub_2116AE0
// Address: 0x2116ae0
//
void __fastcall sub_2116AE0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  char v4; // dl
  __int64 *v5; // rdi
  __int64 *v6; // rsi
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  char v13; // si
  unsigned __int64 v14; // rax
  __int64 *v15; // r9
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 *v18; // rdx
  char v19; // r8
  __int64 v20; // rsi
  __int64 *v21; // rax
  unsigned __int64 v22; // rbx
  __int64 i; // r12
  __int64 v24; // r14
  _QWORD *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r12
  __int64 *v28; // rax
  char v29; // dl
  __int64 v30; // rcx
  __int64 *v31; // rdx
  char v32; // r8
  char v33; // si
  __int64 *v34; // rcx
  unsigned int v35; // r8d
  __int64 *v36; // rsi
  __int64 *v37; // rdi
  unsigned int v38; // r8d
  __int64 *v39; // rcx
  __int64 v40; // rdx
  __int64 *v41; // rsi
  __int64 *v42; // rcx
  char *v43; // [rsp+0h] [rbp-140h]
  __int64 v44; // [rsp+8h] [rbp-138h]
  unsigned __int64 v45; // [rsp+10h] [rbp-130h]
  __int64 v46; // [rsp+18h] [rbp-128h] BYREF
  __int64 v47; // [rsp+20h] [rbp-120h] BYREF
  char v48; // [rsp+30h] [rbp-110h]
  __int64 v49; // [rsp+40h] [rbp-100h]
  __int64 *v50; // [rsp+48h] [rbp-F8h] BYREF
  unsigned __int64 v51; // [rsp+50h] [rbp-F0h]
  char *v52; // [rsp+58h] [rbp-E8h]
  __int64 v53; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v54; // [rsp+68h] [rbp-D8h]
  __int64 v55; // [rsp+70h] [rbp-D0h]
  __int64 v56; // [rsp+78h] [rbp-C8h]
  __int64 *v57; // [rsp+88h] [rbp-B8h]
  __int64 *v58; // [rsp+90h] [rbp-B0h]
  __int64 v59; // [rsp+98h] [rbp-A8h]
  __int64 v60; // [rsp+A0h] [rbp-A0h] BYREF
  _BYTE *v61; // [rsp+A8h] [rbp-98h]
  _BYTE *v62; // [rsp+B0h] [rbp-90h]
  __int64 v63; // [rsp+B8h] [rbp-88h]
  int v64; // [rsp+C0h] [rbp-80h]
  _BYTE v65[120]; // [rsp+C8h] [rbp-78h] BYREF

  v46 = a1;
  v3 = *(__int64 **)(a2 + 8);
  if ( *(__int64 **)(a2 + 16) != v3 )
    goto LABEL_2;
  v40 = *(unsigned int *)(a2 + 28);
  v41 = &v3[v40];
  if ( v3 != v41 )
  {
    v42 = 0;
    do
    {
      if ( a1 == *v3 )
        return;
      if ( *v3 == -2 )
        v42 = v3;
      ++v3;
    }
    while ( v41 != v3 );
    if ( v42 )
    {
      *v42 = a1;
      --*(_DWORD *)(a2 + 32);
      ++*(_QWORD *)a2;
      goto LABEL_6;
    }
  }
  if ( (unsigned int)v40 < *(_DWORD *)(a2 + 24) )
  {
    *(_DWORD *)(a2 + 28) = v40 + 1;
    *v41 = a1;
    ++*(_QWORD *)a2;
  }
  else
  {
LABEL_2:
    sub_16CCBA0(a2, a1);
    if ( !v4 )
      return;
  }
LABEL_6:
  v60 = 0;
  v5 = &v53;
  v6 = &v46;
  v61 = v65;
  v62 = v65;
  v63 = 8;
  v64 = 0;
  sub_21168E0(&v53, &v46, (__int64)&v60);
  v8 = v55;
  v50 = 0;
  v9 = v54;
  v51 = 0;
  v49 = v53;
  v52 = 0;
  v10 = v55 - v54;
  if ( v55 == v54 )
  {
    v5 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_100;
    v11 = sub_22077B0(v55 - v54);
    v8 = v55;
    v9 = v54;
    v5 = (__int64 *)v11;
  }
  v50 = v5;
  v51 = (unsigned __int64)v5;
  v52 = (char *)v5 + v10;
  if ( v9 == v8 )
  {
    v14 = (unsigned __int64)v5;
  }
  else
  {
    v7 = v5;
    v12 = v9;
    do
    {
      if ( v7 )
      {
        *v7 = *(_QWORD *)v12;
        v13 = *(_BYTE *)(v12 + 16);
        *((_BYTE *)v7 + 16) = v13;
        if ( v13 )
          v7[1] = *(_QWORD *)(v12 + 8);
      }
      v12 += 24;
      v7 += 3;
    }
    while ( v12 != v8 );
    v14 = (unsigned __int64)&v5[((unsigned __int64)(v12 - 24 - v9) >> 3) + 3];
  }
  v6 = v58;
  v15 = v57;
  v51 = v14;
  v43 = (char *)((char *)v58 - (char *)v57);
  if ( v58 == v57 )
  {
    v44 = 0;
    goto LABEL_19;
  }
  if ( (unsigned __int64)((char *)v58 - (char *)v57) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_100:
    sub_4261EA(v5, v6, v7);
  v16 = sub_22077B0((char *)v58 - (char *)v57);
  v6 = v58;
  v15 = v57;
  v44 = v16;
  v5 = v50;
  v14 = v51;
LABEL_19:
  if ( v6 == v15 )
  {
    v45 = 0;
  }
  else
  {
    v17 = v44;
    v18 = v15;
    do
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = *v18;
        v19 = *((_BYTE *)v18 + 16);
        *(_BYTE *)(v17 + 16) = v19;
        if ( v19 )
          *(_QWORD *)(v17 + 8) = v18[1];
      }
      v18 += 3;
      v17 += 24;
    }
    while ( v6 != v18 );
    v45 = 8 * ((unsigned __int64)((char *)(v6 - 3) - (char *)v15) >> 3) + 24;
  }
LABEL_26:
  if ( v14 - (_QWORD)v5 == v45 )
    goto LABEL_38;
LABEL_27:
  while ( 2 )
  {
    while ( 2 )
    {
      v20 = *(_QWORD *)(v14 - 24);
      v21 = *(__int64 **)(a2 + 8);
      if ( *(__int64 **)(a2 + 16) != v21 )
        goto LABEL_28;
      v37 = &v21[*(unsigned int *)(a2 + 28)];
      v38 = *(_DWORD *)(a2 + 28);
      if ( v21 != v37 )
      {
        v39 = 0;
        while ( v20 != *v21 )
        {
          if ( *v21 == -2 )
            v39 = v21;
          if ( v37 == ++v21 )
          {
            if ( !v39 )
              goto LABEL_90;
            *v39 = v20;
            --*(_DWORD *)(a2 + 32);
            ++*(_QWORD *)a2;
            goto LABEL_29;
          }
        }
        goto LABEL_29;
      }
LABEL_90:
      if ( v38 < *(_DWORD *)(a2 + 24) )
      {
        *(_DWORD *)(a2 + 28) = v38 + 1;
        *v37 = v20;
        ++*(_QWORD *)a2;
      }
      else
      {
LABEL_28:
        sub_16CCBA0(a2, v20);
      }
LABEL_29:
      v22 = v51;
      while ( 1 )
      {
        if ( *(_BYTE *)(v22 - 8) )
          goto LABEL_31;
        for ( i = *(_QWORD *)(*(_QWORD *)(v22 - 24) + 8LL); i; i = *(_QWORD *)(i + 8) )
        {
          if ( (unsigned __int8)(*((_BYTE *)sub_1648700(i) + 16) - 25) <= 9u )
            break;
        }
        *(_BYTE *)(v22 - 8) = 1;
        *(_QWORD *)(v22 - 16) = i;
LABEL_32:
        if ( i )
          break;
        v51 -= 24LL;
        v5 = v50;
        v22 = v51;
        if ( (__int64 *)v51 == v50 )
        {
          v14 = (unsigned __int64)v50;
          goto LABEL_26;
        }
      }
      v24 = *(_QWORD *)(i + 8);
      for ( *(_QWORD *)(v22 - 16) = v24; v24; *(_QWORD *)(v22 - 16) = v24 )
      {
        if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v24) + 16) - 25) <= 9u )
          break;
        v24 = *(_QWORD *)(v24 + 8);
      }
      v25 = sub_1648700(i);
      v26 = v49;
      v27 = v25[5];
      v28 = *(__int64 **)(v49 + 8);
      if ( *(__int64 **)(v49 + 16) != v28 )
        goto LABEL_36;
      v34 = &v28[*(unsigned int *)(v49 + 28)];
      v35 = *(_DWORD *)(v49 + 28);
      if ( v28 != v34 )
      {
        v36 = 0;
        while ( v27 != *v28 )
        {
          if ( *v28 == -2 )
          {
            v36 = v28;
            if ( v34 == v28 + 1 )
              goto LABEL_63;
            ++v28;
          }
          else if ( v34 == ++v28 )
          {
            if ( !v36 )
              goto LABEL_66;
LABEL_63:
            *v36 = v27;
            --*(_DWORD *)(v26 + 32);
            ++*(_QWORD *)v26;
            goto LABEL_37;
          }
        }
LABEL_31:
        i = *(_QWORD *)(v22 - 16);
        goto LABEL_32;
      }
LABEL_66:
      if ( v35 >= *(_DWORD *)(v49 + 24) )
      {
LABEL_36:
        sub_16CCBA0(v49, v27);
        if ( v29 )
          goto LABEL_37;
        goto LABEL_31;
      }
      *(_DWORD *)(v49 + 28) = v35 + 1;
      *v34 = v27;
      ++*(_QWORD *)v26;
LABEL_37:
      v47 = v27;
      v48 = 0;
      sub_1838D20((unsigned __int64 *)&v50, (__int64)&v47);
      v14 = v51;
      v5 = v50;
      if ( v51 - (_QWORD)v50 != v45 )
        continue;
      break;
    }
LABEL_38:
    if ( v5 != (__int64 *)v14 )
    {
      v30 = v44;
      v31 = v5;
      while ( *v31 == *(_QWORD *)v30 )
      {
        v32 = *((_BYTE *)v31 + 16);
        v33 = *(_BYTE *)(v30 + 16);
        if ( v32 && v33 )
        {
          if ( v31[1] != *(_QWORD *)(v30 + 8) )
            goto LABEL_27;
        }
        else if ( v32 != v33 )
        {
          goto LABEL_27;
        }
        v31 += 3;
        v30 += 24;
        if ( v31 == (__int64 *)v14 )
          goto LABEL_45;
      }
      continue;
    }
    break;
  }
LABEL_45:
  if ( v44 )
  {
    j_j___libc_free_0(v44, v43);
    v5 = v50;
  }
  if ( v5 )
    j_j___libc_free_0(v5, v52 - (char *)v5);
  if ( v57 )
    j_j___libc_free_0(v57, v59 - (_QWORD)v57);
  if ( v54 )
    j_j___libc_free_0(v54, v56 - v54);
  if ( v62 != v61 )
    _libc_free((unsigned __int64)v62);
}
