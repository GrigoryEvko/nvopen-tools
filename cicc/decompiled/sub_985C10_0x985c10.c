// Function: sub_985C10
// Address: 0x985c10
//
__int64 __fastcall sub_985C10(__int64 a1, unsigned __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rax
  char v4; // dl
  int v5; // r12d
  unsigned int v6; // r15d
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 *v12; // r14
  char *v13; // r15
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 *v16; // rbx
  __int64 v17; // r13
  unsigned __int64 v18; // rcx
  unsigned int v20; // r13d
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // r13
  unsigned int v24; // eax
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // rax
  __int64 v27; // r13
  unsigned int v28; // eax
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rax
  __int64 v31; // r13
  unsigned int v32; // eax
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rax
  unsigned int v35; // r13d
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // r13
  unsigned __int64 v38; // rax
  int v39; // eax
  int v40; // eax
  int v41; // eax
  unsigned __int64 v42; // [rsp+8h] [rbp-78h]
  unsigned __int64 v43; // [rsp+8h] [rbp-78h]
  unsigned __int64 v44; // [rsp+10h] [rbp-70h]
  int v45; // [rsp+1Ch] [rbp-64h]
  unsigned int v46; // [rsp+1Ch] [rbp-64h]
  unsigned int v47; // [rsp+1Ch] [rbp-64h]
  unsigned int v48; // [rsp+1Ch] [rbp-64h]
  unsigned int v49; // [rsp+1Ch] [rbp-64h]
  __int64 *v50; // [rsp+20h] [rbp-60h] BYREF
  __int64 v51; // [rsp+28h] [rbp-58h]
  __int64 v52; // [rsp+30h] [rbp-50h] BYREF
  char v53; // [rsp+38h] [rbp-48h] BYREF

  LODWORD(v2) = 0;
  if ( *(_BYTE *)a1 > 0x15u )
    return (unsigned int)v2;
  v50 = &v52;
  v51 = 0x400000000LL;
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_BYTE *)(v3 + 8);
  if ( v4 == 17 )
  {
    v5 = *(_DWORD *)(v3 + 32);
    if ( !v5 )
    {
      LODWORD(v2) = 1;
      return (unsigned int)v2;
    }
    v6 = 0;
    do
    {
      a2 = v6;
      v7 = sub_AD69F0(a1, v6);
      v8 = (unsigned int)v51;
      v9 = (unsigned int)v51 + 1LL;
      if ( v9 > HIDWORD(v51) )
      {
        a2 = (unsigned __int64)&v52;
        sub_C8D5F0(&v50, &v52, v9, 8);
        v8 = (unsigned int)v51;
      }
      ++v6;
      v50[v8] = v7;
      v10 = (unsigned int)(v51 + 1);
      LODWORD(v51) = v51 + 1;
    }
    while ( v5 != v6 );
    v11 = 8 * v10;
    v12 = v50;
    v13 = (char *)&v50[v10];
    v14 = (8 * v10) >> 3;
    v15 = v11 >> 5;
    if ( v15 )
    {
      v16 = v50;
      v2 = &v50[4 * v15];
      while ( 1 )
      {
        v17 = *v16;
        if ( !*v16 || *(_BYTE *)v17 != 17 )
          goto LABEL_15;
        if ( *(_DWORD *)(v17 + 32) > 0x40u )
        {
          v45 = *(_DWORD *)(v17 + 32);
          if ( v45 - (unsigned int)sub_C444A0(v17 + 24) > 0x40 )
            goto LABEL_15;
          v18 = **(_QWORD **)(v17 + 24);
        }
        else
        {
          v18 = *(_QWORD *)(v17 + 24);
        }
        if ( *(_DWORD *)(*(_QWORD *)(v17 + 8) + 8LL) >> 8 <= v18 )
          goto LABEL_15;
        v23 = v16[1];
        a2 = (unsigned __int64)(v16 + 1);
        if ( !v23 || *(_BYTE *)v23 != 17 )
        {
LABEL_32:
          LOBYTE(v2) = v13 == (char *)a2;
          goto LABEL_16;
        }
        v46 = *(_DWORD *)(v23 + 32);
        v24 = *(_DWORD *)(*(_QWORD *)(v23 + 8) + 8LL) >> 8;
        v25 = v24;
        if ( v46 > 0x40 )
        {
          v44 = v24;
          v39 = sub_C444A0(v23 + 24);
          v25 = v44;
          a2 = (unsigned __int64)(v16 + 1);
          if ( v46 - v39 > 0x40 )
            goto LABEL_32;
          v26 = **(_QWORD **)(v23 + 24);
        }
        else
        {
          v26 = *(_QWORD *)(v23 + 24);
        }
        if ( v25 <= v26 )
          goto LABEL_32;
        v27 = v16[2];
        a2 = (unsigned __int64)(v16 + 2);
        if ( !v27 || *(_BYTE *)v27 != 17 )
          goto LABEL_59;
        v47 = *(_DWORD *)(v27 + 32);
        v28 = *(_DWORD *)(*(_QWORD *)(v27 + 8) + 8LL) >> 8;
        v29 = v28;
        if ( v47 > 0x40 )
        {
          v42 = v28;
          v40 = sub_C444A0(v27 + 24);
          a2 = (unsigned __int64)(v16 + 2);
          v29 = v42;
          if ( v47 - v40 > 0x40 )
            goto LABEL_59;
          v30 = **(_QWORD **)(v27 + 24);
        }
        else
        {
          v30 = *(_QWORD *)(v27 + 24);
        }
        if ( v29 <= v30 || (v31 = v16[3], a2 = (unsigned __int64)(v16 + 3), !v31) || *(_BYTE *)v31 != 17 )
        {
LABEL_59:
          LOBYTE(v2) = a2 == (_QWORD)v13;
          goto LABEL_16;
        }
        v48 = *(_DWORD *)(v31 + 32);
        v32 = *(_DWORD *)(*(_QWORD *)(v31 + 8) + 8LL) >> 8;
        v33 = v32;
        if ( v48 > 0x40 )
        {
          v43 = v32;
          v41 = sub_C444A0(v31 + 24);
          a2 = (unsigned __int64)(v16 + 3);
          v33 = v43;
          if ( v48 - v41 > 0x40 )
            goto LABEL_59;
          v34 = **(_QWORD **)(v31 + 24);
        }
        else
        {
          v34 = *(_QWORD *)(v31 + 24);
        }
        if ( v33 <= v34 )
          goto LABEL_59;
        v16 += 4;
        if ( v2 == v16 )
        {
          v14 = (v13 - (char *)v16) >> 3;
          goto LABEL_45;
        }
      }
    }
    v16 = v50;
LABEL_45:
    if ( v14 == 2 )
      goto LABEL_53;
    if ( v14 == 3 )
    {
      v2 = (__int64 *)*v16;
      if ( !*v16 || *(_BYTE *)v2 != 17 )
        goto LABEL_15;
      v35 = *((_DWORD *)v2 + 8);
      if ( v35 > 0x40 )
      {
        if ( v35 - (unsigned int)sub_C444A0(v2 + 3) > 0x40 )
          goto LABEL_15;
        v36 = *(_QWORD *)v2[3];
      }
      else
      {
        v36 = v2[3];
      }
      if ( *(_DWORD *)(v2[1] + 8) >> 8 <= v36 )
        goto LABEL_15;
      ++v16;
LABEL_53:
      v2 = (__int64 *)*v16;
      if ( !*v16 || *(_BYTE *)v2 != 17 )
        goto LABEL_15;
      v49 = *((_DWORD *)v2 + 8);
      v37 = *(_DWORD *)(v2[1] + 8) >> 8;
      if ( v49 > 0x40 )
      {
        if ( v49 - (unsigned int)sub_C444A0(v2 + 3) > 0x40 )
          goto LABEL_15;
        v38 = *(_QWORD *)v2[3];
      }
      else
      {
        v38 = v2[3];
      }
      if ( v37 > v38 )
      {
        ++v16;
        goto LABEL_21;
      }
LABEL_15:
      LOBYTE(v2) = v13 == (char *)v16;
      goto LABEL_16;
    }
    if ( v14 != 1 )
    {
      LODWORD(v2) = 1;
      goto LABEL_16;
    }
  }
  else
  {
    if ( v4 == 18 )
      return (unsigned int)v2;
    v16 = &v52;
    v52 = a1;
    v13 = &v53;
    LODWORD(v51) = 1;
    v12 = &v52;
  }
LABEL_21:
  v2 = (__int64 *)*v16;
  if ( !*v16 || *(_BYTE *)v2 != 17 )
    goto LABEL_15;
  v20 = *((_DWORD *)v2 + 8);
  if ( v20 <= 0x40 )
  {
    v21 = v2[3];
    goto LABEL_25;
  }
  if ( v20 - (unsigned int)sub_C444A0(v2 + 3) > 0x40 )
    goto LABEL_15;
  v21 = *(_QWORD *)v2[3];
LABEL_25:
  v22 = v2[1];
  LODWORD(v2) = 1;
  if ( *(_DWORD *)(v22 + 8) >> 8 <= v21 )
    goto LABEL_15;
LABEL_16:
  if ( v12 != &v52 )
    _libc_free(v12, a2);
  return (unsigned int)v2;
}
