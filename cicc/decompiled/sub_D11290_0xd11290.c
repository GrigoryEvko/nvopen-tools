// Function: sub_D11290
// Address: 0xd11290
//
__int64 __fastcall sub_D11290(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r12
  __int64 v7; // rax
  unsigned __int8 *v9; // rsi
  unsigned __int8 **v10; // r12
  unsigned __int8 **v11; // r15
  char *v12; // rdi
  unsigned __int8 *v13; // r8
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rdx
  unsigned __int64 v17; // r8
  _BYTE *v18; // rsi
  _QWORD *v19; // r12
  __int64 *v20; // r15
  char *v21; // rdi
  unsigned __int8 *v22; // r8
  __int64 v23; // rax
  __int64 v24; // r9
  __int64 v25; // rdx
  unsigned __int64 v26; // r8
  __int64 v27; // r8
  __int64 result; // rax
  _QWORD *v29; // r13
  _QWORD *v30; // r12
  __int64 *v31; // rdi
  __int64 *v32; // r12
  __int64 *v33; // r14
  __int64 v34; // rdx
  __int64 v35; // r15
  char v36; // dl
  __int64 v37; // rdx
  bool v38; // zf
  _BYTE *v39; // r9
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // [rsp+8h] [rbp-118h]
  __int64 v43; // [rsp+10h] [rbp-110h]
  __int64 v45; // [rsp+40h] [rbp-E0h] BYREF
  char *v46; // [rsp+48h] [rbp-D8h]
  int v47; // [rsp+50h] [rbp-D0h]
  char v48; // [rsp+58h] [rbp-C8h] BYREF
  _BYTE *v49; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+68h] [rbp-B8h]
  _BYTE v51[32]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 *v52; // [rsp+90h] [rbp-90h] BYREF
  __int64 v53; // [rsp+98h] [rbp-88h]
  _BYTE v54[32]; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int64 v55; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v56; // [rsp+C8h] [rbp-58h]
  __int64 v57; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v58; // [rsp+D8h] [rbp-48h]

  for ( i = a1[2]; !*(_BYTE *)(i + 24) || *(_QWORD *)(i + 16) != a2; i += 40 )
    ;
  --*(_DWORD *)(*(_QWORD *)(i + 32) + 40LL);
  if ( *(_BYTE *)(i + 24) )
  {
    v7 = *(_QWORD *)(i + 16);
    if ( a3 != v7 )
    {
      if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
        sub_BD60C0((_QWORD *)i);
      *(_QWORD *)(i + 16) = a3;
      if ( a3 != -4096 && a3 != -8192 )
        sub_BD73F0(i);
    }
  }
  else
  {
    *(_QWORD *)i = 6;
    *(_QWORD *)(i + 8) = 0;
    *(_QWORD *)(i + 16) = a3;
    if ( a3 != -8192 && a3 != -4096 )
      sub_BD73F0(i);
    *(_BYTE *)(i + 24) = 1;
  }
  *(_QWORD *)(i + 32) = a4;
  ++*(_DWORD *)(a4 + 40);
  v49 = v51;
  v52 = (__int64 *)v54;
  v9 = (unsigned __int8 *)&v55;
  v50 = 0x400000000LL;
  v53 = 0x400000000LL;
  v56 = 0x400000000LL;
  v55 = (unsigned __int64)&v57;
  sub_E33A00(a2);
  v10 = (unsigned __int8 **)v55;
  v11 = (unsigned __int8 **)(v55 + 8LL * (unsigned int)v56);
  if ( (unsigned __int8 **)v55 != v11 )
  {
    while ( 1 )
    {
      v9 = *v10;
      sub_E33C60(&v45, *v10);
      if ( !v47 && !sub_B491E0(v45) )
        break;
      v12 = v46;
      v13 = *(unsigned __int8 **)(v45
                                + 32 * (*(unsigned int *)v46 - (unsigned __int64)(*(_DWORD *)(v45 + 4) & 0x7FFFFFF)));
      if ( v13 )
        goto LABEL_16;
LABEL_21:
      if ( v12 != &v48 )
        _libc_free(v12, v9);
      if ( v11 == ++v10 )
      {
        v11 = (unsigned __int8 **)v55;
        goto LABEL_29;
      }
    }
    v13 = *(unsigned __int8 **)(v45 - 32);
    if ( v13 )
    {
LABEL_16:
      v9 = sub_BD3990(v13, (__int64)v9);
      if ( !*v9 )
      {
        v14 = sub_D110B0((_QWORD *)*a1, (unsigned __int64)v9);
        v16 = (unsigned int)v50;
        v17 = (unsigned int)v50 + 1LL;
        if ( v17 > HIDWORD(v50) )
        {
          v9 = v51;
          v42 = v14;
          sub_C8D5F0((__int64)&v49, v51, (unsigned int)v50 + 1LL, 8u, v17, v15);
          v16 = (unsigned int)v50;
          v14 = v42;
        }
        *(_QWORD *)&v49[8 * v16] = v14;
        LODWORD(v50) = v50 + 1;
      }
    }
    v12 = v46;
    goto LABEL_21;
  }
LABEL_29:
  if ( v11 != (unsigned __int8 **)&v57 )
    _libc_free(v11, v9);
  v18 = &v55;
  v55 = (unsigned __int64)&v57;
  v56 = 0x400000000LL;
  sub_E33A00(a3);
  v19 = (_QWORD *)v55;
  v20 = (__int64 *)(v55 + 8LL * (unsigned int)v56);
  if ( (__int64 *)v55 != v20 )
  {
    while ( 1 )
    {
      v18 = (_BYTE *)*v19;
      sub_E33C60(&v45, *v19);
      if ( !v47 && !sub_B491E0(v45) )
        break;
      v21 = v46;
      v22 = *(unsigned __int8 **)(v45
                                + 32 * (*(unsigned int *)v46 - (unsigned __int64)(*(_DWORD *)(v45 + 4) & 0x7FFFFFF)));
      if ( v22 )
        goto LABEL_34;
LABEL_39:
      if ( v21 != &v48 )
        _libc_free(v21, v18);
      if ( v20 == ++v19 )
      {
        v20 = (__int64 *)v55;
        goto LABEL_47;
      }
    }
    v22 = *(unsigned __int8 **)(v45 - 32);
    if ( v22 )
    {
LABEL_34:
      v18 = sub_BD3990(v22, (__int64)v18);
      if ( !*v18 )
      {
        v23 = sub_D110B0((_QWORD *)*a1, (unsigned __int64)v18);
        v25 = (unsigned int)v53;
        v26 = (unsigned int)v53 + 1LL;
        if ( v26 > HIDWORD(v53) )
        {
          v18 = v54;
          v43 = v23;
          sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 8u, v26, v24);
          v25 = (unsigned int)v53;
          v23 = v43;
        }
        v52[v25] = v23;
        LODWORD(v53) = v53 + 1;
      }
    }
    v21 = v46;
    goto LABEL_39;
  }
LABEL_47:
  if ( v20 != &v57 )
    _libc_free(v20, v18);
  v27 = (unsigned int)v50;
  result = (unsigned int)v53;
  if ( (unsigned int)v50 == (unsigned __int64)(unsigned int)v53 )
  {
    v31 = v52;
    if ( (_DWORD)v50 )
    {
      v39 = v49;
      v40 = 0;
      do
      {
        v41 = *(_QWORD *)&v39[8 * v40];
        v18 = (_BYTE *)v31[v40];
        for ( result = a1[2]; *(_BYTE *)(result + 24) || *(_QWORD *)(result + 32) != v41; result += 40 )
          ;
        ++v40;
        *(_QWORD *)(result + 32) = v18;
        --*(_DWORD *)(v41 + 40);
        ++*((_DWORD *)v18 + 10);
      }
      while ( v27 != v40 );
    }
  }
  else
  {
    v29 = v49;
    v30 = &v49[8 * (unsigned int)v50];
    if ( v30 != (_QWORD *)v49 )
    {
      do
      {
        v18 = (_BYTE *)*v29++;
        sub_D10670((__int64)a1, (__int64)v18);
      }
      while ( v30 != v29 );
      result = (unsigned int)v53;
    }
    v31 = v52;
    v32 = &v52[result];
    if ( v32 != v52 )
    {
      result = (__int64)&v45;
      v33 = v52;
      while ( 1 )
      {
        v34 = *v33;
        v35 = a1[3];
        v58 = 0;
        v45 = v34;
        if ( v35 == a1[4] )
          break;
        if ( v35 )
        {
          *(_BYTE *)(v35 + 24) = 0;
          if ( (_BYTE)v58 )
          {
            *(_QWORD *)v35 = 6;
            *(_QWORD *)(v35 + 8) = 0;
            v37 = v57;
            v38 = v57 == -4096;
            *(_QWORD *)(v35 + 16) = v57;
            LOBYTE(v18) = !v38;
            if ( v37 != 0 && !v38 && v37 != -8192 )
            {
              v18 = (_BYTE *)(v55 & 0xFFFFFFFFFFFFFFF8LL);
              result = sub_BD6050((unsigned __int64 *)v35, v55 & 0xFFFFFFFFFFFFFFF8LL);
            }
            *(_BYTE *)(v35 + 24) = 1;
            v36 = v58;
            *(_QWORD *)(v35 + 32) = v45;
            a1[3] += 40;
            goto LABEL_68;
          }
          *(_QWORD *)(v35 + 32) = v45;
          a1[3] += 40;
        }
        else
        {
          a1[3] = 40;
        }
LABEL_57:
        ++v33;
        ++*(_DWORD *)(v45 + 40);
        if ( v32 == v33 )
        {
          v31 = v52;
          goto LABEL_62;
        }
      }
      v18 = (_BYTE *)v35;
      result = sub_D10B90(a1 + 2, v35, (__int64)&v55, &v45);
      v36 = v58;
LABEL_68:
      if ( v36 )
      {
        LOBYTE(v58) = 0;
        LOBYTE(v18) = v57 != 0;
        if ( v57 != -4096 && v57 != 0 && v57 != -8192 )
          result = sub_BD60C0(&v55);
      }
      goto LABEL_57;
    }
  }
LABEL_62:
  if ( v31 != (__int64 *)v54 )
    result = _libc_free(v31, v18);
  if ( v49 != v51 )
    return _libc_free(v49, v18);
  return result;
}
