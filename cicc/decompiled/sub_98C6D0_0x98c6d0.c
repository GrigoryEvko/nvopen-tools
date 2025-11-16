// Function: sub_98C6D0
// Address: 0x98c6d0
//
__int64 __fastcall sub_98C6D0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 *v4; // rbx
  __int64 v5; // r13
  _BYTE *v6; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  char *v10; // r15
  __int64 v11; // rax
  char *v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 i; // r13
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  signed __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // r13
  __int64 v32; // r13
  __int64 v33; // rax
  char *v34; // [rsp+8h] [rbp-A8h]
  char *v35; // [rsp+10h] [rbp-A0h]
  _BYTE *v36; // [rsp+10h] [rbp-A0h]
  _BYTE *v37; // [rsp+28h] [rbp-88h]
  _BYTE *v38; // [rsp+28h] [rbp-88h]
  _BYTE *v39; // [rsp+28h] [rbp-88h]
  _BYTE *v40; // [rsp+28h] [rbp-88h]
  _BYTE *v41; // [rsp+28h] [rbp-88h]
  _BYTE *v42; // [rsp+28h] [rbp-88h]
  _BYTE *v43; // [rsp+28h] [rbp-88h]
  __int64 v44; // [rsp+30h] [rbp-80h] BYREF
  __int64 v45; // [rsp+38h] [rbp-78h]
  char *v46; // [rsp+40h] [rbp-70h] BYREF
  __int64 v47; // [rsp+48h] [rbp-68h]
  _BYTE v48[16]; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v49; // [rsp+60h] [rbp-50h] BYREF
  __int64 v50; // [rsp+68h] [rbp-48h]
  _BYTE v51[64]; // [rsp+70h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  v46 = v48;
  v47 = 0x200000000LL;
  v49 = v51;
  v50 = 0x200000000LL;
  if ( !v3 )
    goto LABEL_4;
  v4 = a2;
  do
  {
    v5 = *(_QWORD *)(v3 + 24);
    if ( *(_BYTE *)v5 != 93 )
      goto LABEL_4;
    if ( **(_DWORD **)(v5 + 72) )
    {
      for ( i = *(_QWORD *)(v5 + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v2 = *(_QWORD *)(i + 24);
        if ( *(_BYTE *)v2 == 31 )
        {
          v16 = (unsigned int)v47;
          v17 = (unsigned int)v47 + 1LL;
          if ( v17 > HIDWORD(v47) )
          {
            a2 = (__int64 *)v48;
            sub_C8D5F0(&v46, v48, v17, 8);
            v16 = (unsigned int)v47;
          }
          *(_QWORD *)&v46[8 * v16] = v2;
          LODWORD(v47) = v47 + 1;
        }
      }
    }
    else
    {
      v8 = (unsigned int)v50;
      v9 = (unsigned int)v50 + 1LL;
      if ( v9 > HIDWORD(v50) )
      {
        a2 = (__int64 *)v51;
        sub_C8D5F0(&v49, v51, v9, 8);
        v8 = (unsigned int)v50;
      }
      *(_QWORD *)&v49[8 * v8] = v5;
      LODWORD(v50) = v50 + 1;
    }
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v3 );
  v10 = v46;
  v11 = 8LL * (unsigned int)v47;
  v12 = &v46[v11];
  v13 = v11 >> 5;
  v34 = v12;
  if ( v13 )
  {
    v35 = &v46[32 * v13];
    do
    {
      v14 = *(_QWORD *)(*(_QWORD *)v10 + 40LL);
      v45 = *(_QWORD *)(*(_QWORD *)v10 - 64LL);
      v44 = v14;
      if ( (unsigned __int8)sub_B190C0(&v44) )
      {
        LODWORD(v2) = (_DWORD)v49;
        v6 = &v49[8 * (unsigned int)v50];
        if ( v49 == v6 )
          goto LABEL_23;
        v37 = v49;
        while ( 1 )
        {
          a2 = &v44;
          v2 = *(_QWORD *)v37;
          if ( !(unsigned __int8)sub_B19C20(v4, &v44, *(_QWORD *)(*(_QWORD *)v37 + 40LL)) )
          {
            v2 = *(_QWORD *)(v2 + 16);
            if ( v2 )
              break;
          }
LABEL_21:
          v37 += 8;
          if ( v6 == v37 )
          {
LABEL_22:
            v6 = v49;
            goto LABEL_23;
          }
        }
        while ( 1 )
        {
          a2 = &v44;
          if ( !(unsigned __int8)sub_B19ED0(v4, &v44, v2) )
            break;
          v2 = *(_QWORD *)(v2 + 8);
          if ( !v2 )
            goto LABEL_21;
        }
      }
      v18 = *((_QWORD *)v10 + 1);
      v19 = *(_QWORD *)(v18 - 64);
      v20 = *(_QWORD *)(v18 + 40);
      v45 = v19;
      v44 = v20;
      if ( (unsigned __int8)sub_B190C0(&v44) )
      {
        LODWORD(v2) = (_DWORD)v49;
        v6 = &v49[8 * (unsigned int)v50];
        if ( v49 == v6 )
        {
LABEL_41:
          v10 += 8;
          goto LABEL_23;
        }
        v38 = v49;
        while ( 1 )
        {
          a2 = &v44;
          v2 = *(_QWORD *)v38;
          if ( !(unsigned __int8)sub_B19C20(v4, &v44, *(_QWORD *)(*(_QWORD *)v38 + 40LL)) )
          {
            v2 = *(_QWORD *)(v2 + 16);
            if ( v2 )
              break;
          }
LABEL_39:
          v38 += 8;
          if ( v6 == v38 )
          {
            v6 = v49;
            goto LABEL_41;
          }
        }
        while ( 1 )
        {
          a2 = &v44;
          if ( !(unsigned __int8)sub_B19ED0(v4, &v44, v2) )
            break;
          v2 = *(_QWORD *)(v2 + 8);
          if ( !v2 )
            goto LABEL_39;
        }
      }
      v21 = *((_QWORD *)v10 + 2);
      v22 = *(_QWORD *)(v21 - 64);
      v23 = *(_QWORD *)(v21 + 40);
      v45 = v22;
      v44 = v23;
      if ( (unsigned __int8)sub_B190C0(&v44) )
      {
        LODWORD(v2) = (_DWORD)v49;
        v6 = &v49[8 * (unsigned int)v50];
        if ( v49 == v6 )
        {
LABEL_52:
          LOBYTE(v2) = v34 != v10 + 16;
          goto LABEL_5;
        }
        v39 = v49;
        while ( 1 )
        {
          a2 = &v44;
          v2 = *(_QWORD *)v39;
          if ( !(unsigned __int8)sub_B19C20(v4, &v44, *(_QWORD *)(*(_QWORD *)v39 + 40LL)) )
          {
            v2 = *(_QWORD *)(v2 + 16);
            if ( v2 )
              break;
          }
LABEL_50:
          v39 += 8;
          if ( v6 == v39 )
          {
            v6 = v49;
            goto LABEL_52;
          }
        }
        while ( 1 )
        {
          a2 = &v44;
          if ( !(unsigned __int8)sub_B19ED0(v4, &v44, v2) )
            break;
          v2 = *(_QWORD *)(v2 + 8);
          if ( !v2 )
            goto LABEL_50;
        }
      }
      v24 = *((_QWORD *)v10 + 3);
      v25 = *(_QWORD *)(v24 - 64);
      v26 = *(_QWORD *)(v24 + 40);
      v45 = v25;
      v44 = v26;
      if ( (unsigned __int8)sub_B190C0(&v44) )
      {
        LODWORD(v2) = (_DWORD)v49;
        v6 = &v49[8 * (unsigned int)v50];
        if ( v49 == v6 )
        {
LABEL_63:
          LOBYTE(v2) = v34 != v10 + 24;
          goto LABEL_5;
        }
        v40 = v49;
        while ( 1 )
        {
          a2 = &v44;
          v2 = *(_QWORD *)v40;
          if ( !(unsigned __int8)sub_B19C20(v4, &v44, *(_QWORD *)(*(_QWORD *)v40 + 40LL)) )
          {
            v2 = *(_QWORD *)(v2 + 16);
            if ( v2 )
              break;
          }
LABEL_61:
          v40 += 8;
          if ( v6 == v40 )
          {
            v6 = v49;
            goto LABEL_63;
          }
        }
        while ( 1 )
        {
          a2 = &v44;
          if ( !(unsigned __int8)sub_B19ED0(v4, &v44, v2) )
            break;
          v2 = *(_QWORD *)(v2 + 8);
          if ( !v2 )
            goto LABEL_61;
        }
      }
      v10 += 32;
    }
    while ( v35 != v10 );
  }
  v27 = v34 - v10;
  if ( v34 - v10 != 16 )
  {
    if ( v27 == 24 )
    {
      v33 = *(_QWORD *)(*(_QWORD *)v10 + 40LL);
      v45 = *(_QWORD *)(*(_QWORD *)v10 - 64LL);
      v44 = v33;
      if ( (unsigned __int8)sub_B190C0(&v44) )
      {
        v6 = v49;
        v43 = &v49[8 * (unsigned int)v50];
        if ( v49 == v43 )
          goto LABEL_23;
        while ( 1 )
        {
          v2 = *(_QWORD *)v6;
          a2 = &v44;
          if ( !(unsigned __int8)sub_B19C20(v4, &v44, *(_QWORD *)(*(_QWORD *)v6 + 40LL)) )
          {
            v2 = *(_QWORD *)(v2 + 16);
            if ( v2 )
              break;
          }
LABEL_95:
          v6 += 8;
          if ( v43 == v6 )
            goto LABEL_22;
        }
        while ( 1 )
        {
          a2 = &v44;
          if ( !(unsigned __int8)sub_B19ED0(v4, &v44, v2) )
            break;
          v2 = *(_QWORD *)(v2 + 8);
          if ( !v2 )
            goto LABEL_95;
        }
      }
      v10 += 8;
      goto LABEL_70;
    }
    if ( v27 == 8 )
      goto LABEL_80;
LABEL_4:
    v6 = v49;
    LODWORD(v2) = 0;
    goto LABEL_5;
  }
LABEL_70:
  v28 = *(_QWORD *)(*(_QWORD *)v10 + 40LL);
  v45 = *(_QWORD *)(*(_QWORD *)v10 - 64LL);
  v44 = v28;
  if ( (unsigned __int8)sub_B190C0(&v44) )
  {
    v6 = v49;
    v41 = &v49[8 * (unsigned int)v50];
    if ( v49 == v41 )
      goto LABEL_23;
    while ( 1 )
    {
      v2 = *(_QWORD *)v6;
      a2 = &v44;
      if ( !(unsigned __int8)sub_B19C20(v4, &v44, *(_QWORD *)(*(_QWORD *)v6 + 40LL)) )
      {
        v2 = *(_QWORD *)(v2 + 16);
        if ( v2 )
          break;
      }
LABEL_73:
      v6 += 8;
      if ( v41 == v6 )
        goto LABEL_22;
    }
    while ( 1 )
    {
      a2 = &v44;
      if ( !(unsigned __int8)sub_B19ED0(v4, &v44, v2) )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        goto LABEL_73;
    }
  }
  v10 += 8;
LABEL_80:
  v29 = *(_QWORD *)(*(_QWORD *)v10 + 40LL);
  v45 = *(_QWORD *)(*(_QWORD *)v10 - 64LL);
  v44 = v29;
  v30 = sub_B190C0(&v44);
  v6 = v49;
  LODWORD(v2) = v30;
  if ( !(_BYTE)v30 )
    goto LABEL_5;
  v36 = &v49[8 * (unsigned int)v50];
  if ( v49 == v36 )
  {
LABEL_23:
    LOBYTE(v2) = v34 != v10;
    goto LABEL_5;
  }
  v42 = v49;
  while ( 1 )
  {
    a2 = &v44;
    v31 = *(_QWORD *)v42;
    if ( !(unsigned __int8)sub_B19C20(v4, &v44, *(_QWORD *)(*(_QWORD *)v42 + 40LL)) )
    {
      v32 = *(_QWORD *)(v31 + 16);
      if ( v32 )
        break;
    }
LABEL_84:
    v42 += 8;
    if ( v36 == v42 )
      goto LABEL_22;
  }
  while ( 1 )
  {
    a2 = &v44;
    LODWORD(v2) = sub_B19ED0(v4, &v44, v32);
    if ( !(_BYTE)v2 )
      break;
    v32 = *(_QWORD *)(v32 + 8);
    if ( !v32 )
      goto LABEL_84;
  }
  v6 = v49;
LABEL_5:
  if ( v6 != v51 )
    _libc_free(v6, a2);
  if ( v46 != v48 )
    _libc_free(v46, a2);
  return (unsigned int)v2;
}
