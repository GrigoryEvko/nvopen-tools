// Function: sub_22343F0
// Address: 0x22343f0
//
__int64 *__fastcall sub_22343F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        _DWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        _DWORD *a10)
{
  __int64 *v10; // r12
  void *v11; // rsp
  bool v12; // al
  __int64 v13; // r9
  int v14; // eax
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // r13
  _DWORD **v17; // r12
  __int64 *v18; // rdi
  char v19; // dl
  char v20; // r15
  char v21; // r15
  char v22; // r10
  char v23; // r15
  unsigned __int64 v24; // rax
  __int64 v25; // r10
  unsigned __int64 *v26; // rdx
  int *v27; // rdi
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // rax
  int v32; // eax
  char *v33; // rax
  int v35; // eax
  __int64 v36; // r15
  char v37; // al
  char v38; // dl
  __int64 v39; // rax
  char v40; // al
  void *v41; // rsp
  size_t v42; // rax
  int v43; // eax
  _DWORD *v44; // [rsp+0h] [rbp-60h] BYREF
  char v45; // [rsp+Eh] [rbp-52h]
  char v46; // [rsp+Fh] [rbp-51h]
  __int64 *v47; // [rsp+10h] [rbp-50h] BYREF
  __int64 v48; // [rsp+18h] [rbp-48h]
  __int64 *v49; // [rsp+20h] [rbp-40h] BYREF
  __int64 v50; // [rsp+28h] [rbp-38h]
  __int64 v51; // [rsp+70h] [rbp+10h]
  __int64 v52; // [rsp+70h] [rbp+10h]
  __int64 v53; // [rsp+70h] [rbp+10h]
  __int64 v54; // [rsp+70h] [rbp+10h]
  __int64 v55; // [rsp+70h] [rbp+10h]
  __int64 v56; // [rsp+70h] [rbp+10h]
  __int64 v57; // [rsp+70h] [rbp+10h]

  v44 = a6;
  v49 = (__int64 *)a2;
  v50 = a3;
  v47 = a4;
  v48 = a5;
  v10 = (__int64 *)sub_222F790((_QWORD *)(a9 + 208), a2);
  v11 = alloca(8 * a8 + 8);
  v12 = sub_2233E50((__int64)&v49, (__int64)&v47);
  v13 = a7;
  if ( v12 )
  {
    v14 = v50;
    v15 = 0;
    v16 = 0;
    v17 = 0;
  }
  else
  {
    v16 = 0;
    v36 = 0;
    v37 = sub_2233F00((__int64)&v49);
    v13 = a7;
    v38 = v37;
    v15 = 2 * a8;
    if ( 2 * a8 )
    {
      do
      {
        while ( **(_BYTE **)(v13 + 8 * v36) != v38 )
        {
          v39 = *v10;
          v54 = v13;
          v46 = v38;
          v40 = (*(__int64 (__fastcall **)(__int64 *))(v39 + 16))(v10);
          v38 = v46;
          v13 = v54;
          if ( v46 == v40 )
            break;
          if ( ++v36 == v15 )
            goto LABEL_47;
        }
        *((_DWORD *)&v44 + v16++) = v36++;
      }
      while ( v36 != v15 );
LABEL_47:
      if ( v16 )
      {
        v55 = v13;
        v15 = 0;
        sub_22408B0(v49);
        LODWORD(v50) = -1;
        v13 = v55;
        v41 = alloca(8 * v16 + 8);
        v17 = &v44;
        do
        {
          v56 = v13;
          v42 = strlen(*(const char **)(v13 + 8LL * *((int *)&v44 + v15)));
          v13 = v56;
          (&v44)[v15++] = (_DWORD *)v42;
        }
        while ( v16 != v15 );
        v14 = -1;
        v16 = 1;
      }
      else
      {
        v14 = v50;
        v15 = 0;
        v17 = 0;
      }
    }
    else
    {
      v14 = v50;
      v17 = 0;
    }
  }
  v18 = v49;
LABEL_4:
  v19 = v14 == -1;
  v20 = v19 & (v18 != 0);
  if ( v20 )
  {
    v19 = 0;
    if ( v18[2] >= (unsigned __int64)v18[3] )
    {
      v31 = *v18;
      v52 = v13;
      v46 = 0;
      v32 = (*(__int64 (__fastcall **)(__int64 *))(v31 + 72))(v18);
      v19 = v46;
      v13 = v52;
      if ( v32 == -1 )
      {
        v49 = 0;
        v19 = v20;
      }
    }
  }
  v21 = (_DWORD)v48 == -1;
  v22 = v21 & (v47 != 0);
  if ( v22 )
  {
    v21 = 0;
    if ( v47[2] >= (unsigned __int64)v47[3] )
    {
      v29 = *v47;
      v51 = v13;
      v45 = v22;
      v46 = v19;
      v30 = (*(__int64 (**)(void))(v29 + 72))();
      v19 = v46;
      v13 = v51;
      if ( v30 == -1 )
      {
        v47 = 0;
        v21 = v45;
      }
    }
  }
  if ( v19 != v21 )
  {
    v23 = v50;
    if ( (_DWORD)v50 == -1 && v49 )
    {
      v33 = (char *)v49[2];
      if ( (unsigned __int64)v33 < v49[3] )
      {
        v23 = *v33;
        if ( !v15 )
          goto LABEL_29;
        goto LABEL_10;
      }
      v57 = v13;
      v43 = (*(__int64 (**)(void))(*v49 + 72))();
      v13 = v57;
      v23 = v43;
      if ( v43 == -1 )
        v49 = 0;
    }
    if ( !v15 )
      goto LABEL_29;
LABEL_10:
    v24 = 0;
    v25 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v26 = (unsigned __int64 *)&v17[v24];
        if ( *v26 > v16 )
          break;
        ++v25;
        ++v24;
LABEL_12:
        if ( v15 <= v24 )
          goto LABEL_16;
      }
      v27 = (int *)&v44 + v24;
      if ( *(_BYTE *)(*(_QWORD *)(v13 + 8LL * *v27) + v16) == v23 )
      {
        ++v24;
        goto LABEL_12;
      }
      --v15;
      *v27 = *((_DWORD *)&v44 + v15);
      *v26 = (unsigned __int64)v17[v15];
      if ( v15 <= v24 )
      {
LABEL_16:
        if ( v25 == v15 )
          break;
        v18 = v49;
        v28 = v49[2];
        if ( v28 >= v49[3] )
        {
          v53 = v13;
          (*(void (__fastcall **)(__int64 *))(*v49 + 80))(v49);
          v18 = v49;
          v13 = v53;
        }
        else
        {
          v49[2] = v28 + 1;
        }
        LODWORD(v50) = -1;
        ++v16;
        v14 = -1;
        goto LABEL_4;
      }
    }
  }
  if ( v15 != 1 )
  {
    if ( v15 == 2 && (*v17 == (_DWORD *)v16 || v17[1] == (_DWORD *)v16) )
      goto LABEL_34;
LABEL_29:
    *a10 |= 4u;
    return v49;
  }
  if ( *v17 != (_DWORD *)v16 )
    goto LABEL_29;
LABEL_34:
  v35 = (int)v44;
  if ( (int)v44 >= (int)a8 )
    v35 = (_DWORD)v44 - a8;
  *v44 = v35;
  return v49;
}
