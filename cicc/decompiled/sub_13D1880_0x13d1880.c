// Function: sub_13D1880
// Address: 0x13d1880
//
__int64 __fastcall sub_13D1880(_BYTE *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  _BYTE *v4; // r14
  __int64 v6; // r13
  __int64 v8; // r10
  unsigned int *v9; // rdx
  char v10; // bl
  unsigned int *v11; // rax
  char v12; // si
  unsigned int v13; // edx
  unsigned __int8 v14; // al
  unsigned __int8 v15; // dl
  __int64 v16; // rax
  unsigned int *v17; // rdi
  _BYTE *v18; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int *v23; // r9
  __int64 v24; // rsi
  __int64 v25; // rdx
  unsigned int *v26; // rax
  unsigned int *v27; // rdx
  __int64 v28; // r15
  __int64 v29; // rbx
  _BYTE *v30; // r13
  _BYTE *v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r15
  __int64 v34; // rsi
  _BYTE *v35; // rdi
  int v36; // r14d
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // eax
  bool v40; // r9
  int *v41; // rax
  int *v42; // rdi
  int v43; // edx
  int v44; // esi
  _BYTE *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // [rsp-F0h] [rbp-F0h]
  __int64 v48; // [rsp-E8h] [rbp-E8h]
  __int64 v49; // [rsp-E8h] [rbp-E8h]
  _BYTE *v51; // [rsp-E0h] [rbp-E0h]
  __int64 v52; // [rsp-E0h] [rbp-E0h]
  __int64 v53; // [rsp-D8h] [rbp-D8h]
  __int64 v54; // [rsp-D8h] [rbp-D8h]
  __int64 v55; // [rsp-D8h] [rbp-D8h]
  __int64 v56; // [rsp-D8h] [rbp-D8h]
  unsigned int *v57; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v58; // [rsp-C0h] [rbp-C0h]
  _BYTE v59[184]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( a3[16] == 9 )
    return sub_1599EF0(a4);
  v4 = a1;
  v6 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
  v48 = *(_QWORD *)a1;
  v47 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v57 = (unsigned int *)v59;
  v58 = 0x2000000000LL;
  sub_15FAA20(a3, &v57);
  v8 = a4;
  if ( (_DWORD)v6 )
  {
    v9 = v57;
    v10 = 0;
    v11 = v57 + 1;
    v12 = 0;
    while ( 1 )
    {
      v13 = *v9;
      if ( v13 != -1 )
      {
        if ( (unsigned int)v47 <= v13 )
          v10 = 1;
        else
          v12 = 1;
      }
      v9 = v11;
      if ( v11 == &v57[(unsigned int)(v6 - 1) + 1] )
        break;
      ++v11;
    }
    if ( v12 )
    {
      if ( v10 )
        goto LABEL_12;
    }
    else
    {
      v54 = a2;
      v38 = sub_1599EF0(v48);
      v8 = a4;
      a2 = v54;
      v4 = (_BYTE *)v38;
      if ( v10 )
      {
LABEL_12:
        v14 = v4[16];
        v15 = *(_BYTE *)(a2 + 16);
        if ( v14 <= 0x10u )
          goto LABEL_13;
LABEL_22:
        if ( v15 > 0x10u )
          goto LABEL_24;
        goto LABEL_23;
      }
    }
  }
  else
  {
    v20 = sub_1599EF0(v48);
    v8 = a4;
    v4 = (_BYTE *)v20;
  }
  v53 = v8;
  v21 = sub_1599EF0(v48);
  v8 = v53;
  a2 = v21;
  v14 = v4[16];
  v15 = *(_BYTE *)(a2 + 16);
  if ( v14 > 0x10u )
    goto LABEL_22;
LABEL_13:
  if ( v15 > 0x10u )
  {
    v41 = (int *)v57;
    v42 = (int *)&v57[(unsigned int)v58];
    if ( v57 == (unsigned int *)v42 )
    {
      v14 = v15;
    }
    else
    {
      do
      {
        v43 = *v41;
        if ( *v41 != -1 )
        {
          v44 = v47 + v43;
          if ( v43 >= (int)v47 )
            v44 = v43 - v47;
          *v41 = v44;
        }
        ++v41;
      }
      while ( v42 != v41 );
      v14 = *(_BYTE *)(a2 + 16);
    }
    v45 = v4;
    v4 = (_BYTE *)a2;
    a2 = (__int64)v45;
LABEL_23:
    if ( v14 == 85 && *(_BYTE *)(a2 + 16) == 9 && v8 == v48 )
    {
      v52 = v8;
      v56 = a2;
      v46 = sub_15A1020(*((_QWORD *)v4 - 3));
      a2 = v56;
      v8 = v52;
      if ( v46 )
      {
        v17 = v57;
        v18 = v4;
        goto LABEL_15;
      }
    }
LABEL_24:
    v17 = v57;
    v22 = 4LL * (unsigned int)v58;
    v23 = &v57[(unsigned __int64)v22 / 4];
    v24 = v22 >> 2;
    v25 = v22 >> 4;
    if ( v25 )
    {
      v26 = v57;
      v27 = &v57[4 * v25];
      while ( *v26 != -1 )
      {
        if ( v26[1] == -1 )
        {
          ++v26;
          break;
        }
        if ( v26[2] == -1 )
        {
          v26 += 2;
          break;
        }
        if ( v26[3] == -1 )
        {
          v26 += 3;
          break;
        }
        v26 += 4;
        if ( v27 == v26 )
        {
          v24 = v23 - v26;
          goto LABEL_62;
        }
      }
LABEL_31:
      v18 = 0;
      if ( v23 != v26 )
        goto LABEL_15;
LABEL_32:
      if ( (_DWORD)v6 )
      {
        v28 = (unsigned int)v6;
        v29 = 0;
        v30 = 0;
        v31 = v4;
        v32 = v28;
        v33 = v8;
        while ( 1 )
        {
          v34 = v17[v29];
          if ( (_DWORD)v34 == -1 )
            break;
          v35 = (_BYTE *)a2;
          v18 = v31;
          v36 = 2;
          while ( 1 )
          {
            v37 = *(_QWORD *)(*(_QWORD *)v18 + 32LL);
            if ( (int)v37 <= (int)v34 )
            {
              v34 = (unsigned int)(v34 - v37);
              v18 = v35;
            }
            if ( v18[16] != 85 )
              break;
            v49 = a2;
            v51 = v31;
            v55 = v32;
            v39 = sub_15FA9D0(*((_QWORD *)v18 - 3), v34);
            v35 = (_BYTE *)*((_QWORD *)v18 - 6);
            v32 = v55;
            v40 = v36 == 0;
            v34 = v39;
            v18 = (_BYTE *)*((_QWORD *)v18 - 9);
            --v36;
            v31 = v51;
            a2 = v49;
            if ( v39 == -1 || v40 )
            {
              v17 = v57;
              v18 = 0;
              goto LABEL_15;
            }
          }
          v17 = v57;
          if ( v30 )
          {
            if ( v18 != v30 )
              break;
          }
          if ( (_DWORD)v34 != (_DWORD)v29 || v33 != *(_QWORD *)v18 )
            break;
          if ( ++v29 == v32 )
            goto LABEL_15;
          v30 = v18;
        }
      }
      v18 = 0;
      goto LABEL_15;
    }
    v26 = v57;
LABEL_62:
    if ( v24 != 2 )
    {
      if ( v24 != 3 )
      {
        if ( v24 != 1 )
          goto LABEL_32;
        goto LABEL_65;
      }
      if ( *v26 == -1 )
        goto LABEL_31;
      ++v26;
    }
    if ( *v26 == -1 )
      goto LABEL_31;
    ++v26;
LABEL_65:
    if ( *v26 != -1 )
      goto LABEL_32;
    goto LABEL_31;
  }
  v16 = sub_1584900(v4, a2, a3);
  v17 = v57;
  v18 = (_BYTE *)v16;
LABEL_15:
  if ( v17 != (unsigned int *)v59 )
    _libc_free((unsigned __int64)v17);
  return (__int64)v18;
}
