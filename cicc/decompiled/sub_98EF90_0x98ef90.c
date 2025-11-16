// Function: sub_98EF90
// Address: 0x98ef90
//
__int64 __fastcall sub_98EF90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rsi
  __int64 v8; // r12
  __int64 **v9; // rdx
  char *v10; // rcx
  __int64 v11; // rbx
  char *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rbx
  char *v15; // rbx
  __int64 *v16; // r8
  char *v17; // rax
  char *v18; // rax
  char *v19; // rdx
  _QWORD *v20; // rdi
  unsigned int v21; // r14d
  __int64 *v23; // r8
  __int64 v24; // r14
  __int64 *v25; // r8
  char *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdx
  char v29; // dl
  __int64 i; // rbx
  __int64 v31; // r12
  unsigned int v32; // eax
  char v33; // r9
  char v34; // al
  char *v38; // [rsp+28h] [rbp-168h]
  _QWORD *v39; // [rsp+30h] [rbp-160h] BYREF
  __int64 v40; // [rsp+38h] [rbp-158h]
  _QWORD v41[16]; // [rsp+40h] [rbp-150h] BYREF
  __int64 v42; // [rsp+C0h] [rbp-D0h] BYREF
  char *v43; // [rsp+C8h] [rbp-C8h]
  __int64 v44; // [rsp+D0h] [rbp-C0h]
  int v45; // [rsp+D8h] [rbp-B8h]
  char v46; // [rsp+DCh] [rbp-B4h]
  char v47; // [rsp+E0h] [rbp-B0h] BYREF

  v5 = (unsigned __int8 *)v41;
  v41[0] = a1;
  v42 = 0;
  v44 = 16;
  v45 = 0;
  v46 = 1;
  v39 = v41;
  v43 = &v47;
  v40 = 0x1000000001LL;
  LODWORD(v6) = 1;
  while ( 1 )
  {
    v7 = &v42;
    v8 = *(_QWORD *)&v5[8 * (unsigned int)v6 - 8];
    LODWORD(v40) = v6 - 1;
    if ( (unsigned __int8)sub_98D5C0((unsigned __int8 *)v8, (__int64)&v42, v5, (unsigned int)v6, a5) )
    {
      v7 = (__int64 *)v8;
      v32 = sub_B19DB0(a3, v8, a2);
      if ( (_BYTE)v32 )
        break;
    }
    if ( a1 == v8 )
      goto LABEL_15;
    v11 = 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
    {
      v12 = *(char **)(v8 - 8);
      v38 = &v12[v11];
    }
    else
    {
      v38 = (char *)v8;
      v12 = (char *)(v8 - v11);
    }
    v13 = v11 >> 5;
    v14 = v11 >> 7;
    if ( v14 )
    {
      v15 = &v12[128 * v14];
      while ( 1 )
      {
        v16 = *(__int64 **)v12;
        if ( v46 )
          break;
        v7 = *(__int64 **)v12;
        if ( sub_C8CA60(&v42, v16, v9, v10, v16) )
          goto LABEL_13;
LABEL_29:
        v23 = (__int64 *)*((_QWORD *)v12 + 4);
        v24 = (__int64)(v12 + 32);
        if ( !v46 )
        {
          v7 = (__int64 *)*((_QWORD *)v12 + 4);
          if ( !sub_C8CA60(&v42, v23, v9, v10, v23) )
            goto LABEL_37;
          goto LABEL_34;
        }
        v7 = (__int64 *)v43;
        v17 = &v43[8 * HIDWORD(v44)];
        v10 = v43;
        if ( v43 == v17 )
          goto LABEL_64;
LABEL_31:
        v9 = (__int64 **)v7;
        while ( *v9 != v23 )
        {
          if ( ++v9 == (__int64 **)v17 )
            goto LABEL_64;
        }
LABEL_34:
        if ( (unsigned __int8)sub_98D0D0(v24, (__int64)v7, (__int64)v9, (__int64)v10, (unsigned int)v23) )
          goto LABEL_35;
LABEL_37:
        v25 = (__int64 *)*((_QWORD *)v12 + 8);
        v24 = (__int64)(v12 + 64);
        if ( v46 )
        {
          v7 = (__int64 *)v43;
          v17 = &v43[8 * HIDWORD(v44)];
          v10 = v43;
          if ( v43 == v17 )
            goto LABEL_65;
LABEL_39:
          v9 = (__int64 **)v7;
          while ( v25 != *v9 )
          {
            if ( ++v9 == (__int64 **)v17 )
              goto LABEL_65;
          }
LABEL_42:
          if ( (unsigned __int8)sub_98D0D0(v24, (__int64)v7, (__int64)v9, (__int64)v10, (unsigned int)v25) )
            goto LABEL_35;
          goto LABEL_43;
        }
        v7 = (__int64 *)*((_QWORD *)v12 + 8);
        if ( sub_C8CA60(&v42, v25, v9, v10, v25) )
          goto LABEL_42;
LABEL_43:
        a5 = *((_QWORD *)v12 + 12);
        v24 = (__int64)(v12 + 96);
        if ( !v46 )
        {
          v7 = (__int64 *)*((_QWORD *)v12 + 12);
          if ( sub_C8CA60(&v42, a5, v9, v10, a5) )
            goto LABEL_49;
          goto LABEL_50;
        }
        v7 = (__int64 *)v43;
        v10 = v43;
        v17 = &v43[8 * HIDWORD(v44)];
LABEL_45:
        if ( v7 != (__int64 *)v17 )
        {
          while ( a5 != *(_QWORD *)v10 )
          {
            v10 += 8;
            if ( v10 == v17 )
              goto LABEL_50;
          }
LABEL_49:
          if ( (unsigned __int8)sub_98D0D0(v24, (__int64)v7, (__int64)v9, (__int64)v10, a5) )
          {
LABEL_35:
            v12 = (char *)v24;
            goto LABEL_14;
          }
        }
LABEL_50:
        v12 += 128;
        if ( v15 == v12 )
        {
          v13 = (v38 - v12) >> 5;
          goto LABEL_52;
        }
      }
      v7 = (__int64 *)v43;
      v17 = &v43[8 * HIDWORD(v44)];
      v10 = v43;
      if ( v43 == v17 )
      {
LABEL_63:
        v23 = (__int64 *)*((_QWORD *)v12 + 4);
        v24 = (__int64)(v12 + 32);
        if ( v43 == v17 )
        {
LABEL_64:
          v25 = (__int64 *)*((_QWORD *)v12 + 8);
          v24 = (__int64)(v12 + 64);
          if ( v7 == (__int64 *)v17 )
          {
LABEL_65:
            a5 = *((_QWORD *)v12 + 12);
            v24 = (__int64)(v12 + 96);
            goto LABEL_45;
          }
          goto LABEL_39;
        }
        goto LABEL_31;
      }
      v9 = (__int64 **)v43;
      while ( v16 != *v9 )
      {
        if ( v17 == (char *)++v9 )
          goto LABEL_63;
      }
LABEL_13:
      if ( (unsigned __int8)sub_98D0D0((__int64)v12, (__int64)v7, (__int64)v9, (__int64)v10, (unsigned int)v16) )
        goto LABEL_14;
      goto LABEL_29;
    }
LABEL_52:
    if ( v13 == 2 )
    {
      v7 = *(__int64 **)v12;
      if ( v46 )
        goto LABEL_95;
      goto LABEL_89;
    }
    if ( v13 != 3 )
    {
      if ( v13 != 1 )
        goto LABEL_20;
      v7 = *(__int64 **)v12;
      if ( !v46 )
        goto LABEL_92;
      goto LABEL_56;
    }
    v33 = v46;
    v7 = *(__int64 **)v12;
    if ( v46 )
    {
      v26 = v43;
      v27 = HIDWORD(v44);
      v10 = &v43[8 * HIDWORD(v44)];
      v9 = (__int64 **)v43;
      if ( v43 == v10 )
      {
        v7 = (__int64 *)*((_QWORD *)v12 + 4);
        v12 += 32;
LABEL_96:
        v10 = &v26[8 * v27];
        v9 = (__int64 **)v26;
        if ( v26 == v10 )
        {
          v7 = (__int64 *)*((_QWORD *)v12 + 4);
          v12 += 32;
          goto LABEL_57;
        }
        while ( *v9 != v7 )
        {
          if ( v10 == (char *)++v9 )
          {
            v34 = 1;
            goto LABEL_91;
          }
        }
        goto LABEL_100;
      }
      while ( v7 != *v9 )
      {
        if ( v10 == (char *)++v9 )
          goto LABEL_88;
      }
    }
    else if ( !sub_C8CA60(&v42, v7, v9, v10, a5) )
    {
      goto LABEL_87;
    }
    if ( (unsigned __int8)sub_98D0D0((__int64)v12, (__int64)v7, (__int64)v9, (__int64)v10, a5) )
      goto LABEL_14;
LABEL_87:
    v33 = v46;
LABEL_88:
    v12 += 32;
    v7 = *(__int64 **)v12;
    if ( v33 )
    {
LABEL_95:
      v26 = v43;
      v27 = HIDWORD(v44);
      goto LABEL_96;
    }
LABEL_89:
    if ( !sub_C8CA60(&v42, v7, v9, v10, a5) )
      goto LABEL_90;
LABEL_100:
    if ( !(unsigned __int8)sub_98D0D0((__int64)v12, (__int64)v7, (__int64)v9, (__int64)v10, a5) )
    {
LABEL_90:
      v34 = v46;
LABEL_91:
      v12 += 32;
      v7 = *(__int64 **)v12;
      if ( !v34 )
      {
LABEL_92:
        if ( !sub_C8CA60(&v42, v7, v9, v10, a5) )
          goto LABEL_20;
        goto LABEL_61;
      }
LABEL_56:
      v26 = v43;
      v27 = HIDWORD(v44);
LABEL_57:
      v28 = (__int64)&v26[8 * v27];
      if ( v26 == (char *)v28 )
        goto LABEL_20;
      while ( *(__int64 **)v26 != v7 )
      {
        v26 += 8;
        if ( (char *)v28 == v26 )
          goto LABEL_20;
      }
LABEL_61:
      if ( !(unsigned __int8)sub_98D0D0((__int64)v12, (__int64)v7, v28, (__int64)v10, a5) )
        goto LABEL_20;
    }
LABEL_14:
    if ( v12 == v38 )
      goto LABEL_20;
LABEL_15:
    if ( v46 )
    {
      v18 = v43;
      v19 = &v43[8 * HIDWORD(v44)];
      if ( v43 != v19 )
      {
        while ( v8 != *(_QWORD *)v18 )
        {
          v18 += 8;
          if ( v19 == v18 )
            goto LABEL_79;
        }
LABEL_20:
        LODWORD(v6) = v40;
        goto LABEL_21;
      }
LABEL_79:
      if ( HIDWORD(v44) < (unsigned int)v44 )
      {
        ++HIDWORD(v44);
        *(_QWORD *)v19 = v8;
        v6 = (unsigned int)v40;
        ++v42;
LABEL_71:
        for ( i = *(_QWORD *)(v8 + 16); i; i = *(_QWORD *)(i + 8) )
        {
          v31 = *(_QWORD *)(i + 24);
          if ( v6 + 1 > (unsigned __int64)HIDWORD(v40) )
          {
            v7 = v41;
            sub_C8D5F0(&v39, v41, v6 + 1, 8);
            v6 = (unsigned int)v40;
          }
          v39[v6] = v31;
          v6 = (unsigned int)(v40 + 1);
          LODWORD(v40) = v40 + 1;
        }
        goto LABEL_21;
      }
    }
    v7 = (__int64 *)v8;
    sub_C8CC70(&v42, v8);
    v6 = (unsigned int)v40;
    if ( v29 )
      goto LABEL_71;
LABEL_21:
    v20 = v39;
    v5 = (unsigned __int8 *)v39;
    if ( !(_DWORD)v6 )
    {
      v21 = 0;
      goto LABEL_23;
    }
  }
  v20 = v39;
  v21 = v32;
LABEL_23:
  if ( v20 != v41 )
    _libc_free(v20, v7);
  if ( !v46 )
    _libc_free(v43, v7);
  return v21;
}
