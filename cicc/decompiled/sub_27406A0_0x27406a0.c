// Function: sub_27406A0
// Address: 0x27406a0
//
__int64 __fastcall sub_27406A0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 *v11; // rbx
  _QWORD *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rdx
  unsigned int v15; // r10d
  char *v16; // r10
  __int64 *v17; // r15
  __int16 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rax
  char *v22; // rax
  unsigned int v23; // eax
  char **v24; // rbx
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rsi
  __int64 v29; // rdx
  _QWORD *v30; // rdi
  _BYTE *v31; // rdi
  unsigned __int64 v33; // rdx
  char *v34; // rax
  __int64 v35; // r8
  char *v36; // rbx
  __int64 v37; // [rsp+0h] [rbp-130h]
  __int64 v38; // [rsp+8h] [rbp-128h]
  char *v39; // [rsp+10h] [rbp-120h]
  char *v40; // [rsp+18h] [rbp-118h]
  char *v41; // [rsp+18h] [rbp-118h]
  char *v42; // [rsp+18h] [rbp-118h]
  char *v43; // [rsp+18h] [rbp-118h]
  char *v44; // [rsp+20h] [rbp-110h] BYREF
  __int64 v45; // [rsp+28h] [rbp-108h]
  _BYTE v46[64]; // [rsp+30h] [rbp-100h] BYREF
  _BYTE *v47; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v48; // [rsp+78h] [rbp-B8h]
  _BYTE v49[176]; // [rsp+80h] [rbp-B0h] BYREF

  v6 = a3;
  v7 = a2 + 1;
  v8 = 8 * a3 - 8;
  v9 = v8 >> 3;
  v11 = a2;
  v12 = (__int64 *)((char *)a2 + v8 + 8);
  v13 = v8 >> 5;
  if ( v13 > 0 )
  {
    v14 = (__int64)&v11[4 * v13 + 1];
    while ( !*v7 )
    {
      if ( v7[1] )
      {
        ++v7;
        goto LABEL_8;
      }
      if ( v7[2] )
      {
        v7 += 2;
        goto LABEL_8;
      }
      if ( v7[3] )
      {
        v7 += 3;
        goto LABEL_8;
      }
      v7 += 4;
      if ( (_QWORD *)v14 == v7 )
      {
        v9 = v12 - v7;
        goto LABEL_35;
      }
    }
    goto LABEL_8;
  }
LABEL_35:
  if ( v9 == 2 )
    goto LABEL_42;
  if ( v9 == 3 )
  {
    if ( *v7 )
      goto LABEL_8;
    ++v7;
LABEL_42:
    if ( *v7 )
      goto LABEL_8;
    ++v7;
    goto LABEL_38;
  }
  v15 = 0;
  if ( v9 != 1 )
    return v15;
LABEL_38:
  v15 = 0;
  if ( !*v7 )
    return v15;
LABEL_8:
  v15 = 0;
  if ( v12 == v7 )
    return v15;
  v16 = v46;
  v17 = &v11[v6];
  v44 = v46;
  v45 = 0x400000000LL;
  if ( v17 == v11 )
  {
    v19 = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)v19 )
    {
      v24 = &v47;
      v48 = 0x800000000LL;
      v23 = v19;
      v47 = v49;
      goto LABEL_21;
    }
  }
  else
  {
    v18 = 0;
    v19 = 0;
    do
    {
      v20 = *v11;
      if ( *v11 )
      {
        v9 = HIDWORD(v45);
        v21 = (unsigned int)v19;
        if ( (unsigned int)v19 >= (unsigned __int64)HIDWORD(v45) )
        {
          v33 = (unsigned int)v19 + 1LL;
          LOWORD(a6) = v18;
          if ( HIDWORD(v45) < (unsigned __int64)(v21 + 1) )
          {
            v37 = a6;
            v38 = v6;
            v39 = v16;
            sub_C8D5F0((__int64)&v44, v16, v33, 0x10u, v6, a6);
            v21 = (unsigned int)v45;
            a6 = v37;
            v6 = v38;
            v16 = v39;
          }
          v34 = &v44[16 * v21];
          *(_QWORD *)v34 = v20;
          *((_QWORD *)v34 + 1) = a6;
          v19 = (unsigned int)(v45 + 1);
          LODWORD(v45) = v45 + 1;
        }
        else
        {
          v22 = &v44[16 * (unsigned int)v19];
          if ( v22 )
          {
            *(_QWORD *)v22 = v20;
            *((_WORD *)v22 + 4) = v18;
            LODWORD(v19) = v45;
          }
          v19 = (unsigned int)(v19 + 1);
          LODWORD(v45) = v19;
        }
      }
      ++v11;
      ++v18;
    }
    while ( v17 != v11 );
    v23 = *(_DWORD *)(a1 + 16);
    if ( v23 )
      goto LABEL_19;
  }
  *(_QWORD *)a1 = v6;
  v23 = 0;
LABEL_19:
  v24 = &v47;
  v47 = v49;
  v48 = 0x800000000LL;
  if ( (_DWORD)v19 )
  {
    v41 = v16;
    sub_27389D0((__int64)&v47, &v44, v19, v9, v6, a6);
    v23 = *(_DWORD *)(a1 + 16);
    v16 = v41;
  }
LABEL_21:
  v25 = v23;
  v26 = *(unsigned int *)(a1 + 20);
  v27 = *(_QWORD *)(a1 + 8);
  v28 = v23 + 1LL;
  if ( v28 > v26 )
  {
    v35 = a1 + 8;
    if ( v27 > (unsigned __int64)&v47 )
    {
      v43 = v16;
    }
    else
    {
      v43 = v16;
      if ( (unsigned __int64)&v47 < v27 + 144LL * v23 )
      {
        v36 = (char *)&v47 - v27;
        sub_2740590(a1 + 8, v28, v23, v26, v35, a6);
        v27 = *(_QWORD *)(a1 + 8);
        v25 = *(unsigned int *)(a1 + 16);
        v16 = v43;
        v24 = (char **)&v36[v27];
        v23 = *(_DWORD *)(a1 + 16);
        goto LABEL_22;
      }
    }
    sub_2740590(a1 + 8, v28, v23, v26, v35, a6);
    v25 = *(unsigned int *)(a1 + 16);
    v27 = *(_QWORD *)(a1 + 8);
    v16 = v43;
    v23 = *(_DWORD *)(a1 + 16);
  }
LABEL_22:
  v29 = 144 * v25;
  v30 = (_QWORD *)(v29 + v27);
  if ( v30 )
  {
    *v30 = v30 + 2;
    v30[1] = 0x800000000LL;
    if ( *((_DWORD *)v24 + 2) )
    {
      v42 = v16;
      sub_27389D0((__int64)v30, v24, v29, v26, v6, a6);
      v23 = *(_DWORD *)(a1 + 16);
      v16 = v42;
    }
    else
    {
      v23 = *(_DWORD *)(a1 + 16);
    }
  }
  v31 = v47;
  *(_DWORD *)(a1 + 16) = v23 + 1;
  if ( v31 != v49 )
  {
    v40 = v16;
    _libc_free((unsigned __int64)v31);
    v16 = v40;
  }
  if ( v44 != v16 )
    _libc_free((unsigned __int64)v44);
  return 1;
}
