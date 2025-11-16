// Function: sub_185A320
// Address: 0x185a320
//
__int64 __fastcall sub_185A320(__int64 a1, void *a2)
{
  int v4; // edi
  void **v5; // rsi
  unsigned int v6; // ecx
  void **v7; // r14
  void *v8; // r8
  _QWORD *v9; // rax
  void *v10; // rdi
  __int64 v11; // rdx
  _QWORD **v12; // r14
  _QWORD **i; // r13
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r15
  __int64 v17; // rdi
  _QWORD *v18; // rbx
  _QWORD *v19; // r13
  __int64 v20; // rdi
  _QWORD **v21; // r14
  _QWORD **j; // r13
  __int64 v23; // rax
  _QWORD *v24; // rbx
  _QWORD *v25; // r15
  __int64 v26; // rdi
  _QWORD *v27; // rbx
  _QWORD *v28; // r13
  __int64 v29; // rdi
  void **v31; // rax
  void **v32; // rbx
  void **v33; // rdx
  __int64 v34; // [rsp+0h] [rbp-180h] BYREF
  _QWORD *v35; // [rsp+8h] [rbp-178h]
  __int64 v36; // [rsp+10h] [rbp-170h]
  __int64 v37; // [rsp+18h] [rbp-168h]
  __int64 v38; // [rsp+20h] [rbp-160h]
  __int64 v39; // [rsp+28h] [rbp-158h]
  __int64 v40; // [rsp+30h] [rbp-150h]
  __int64 v41; // [rsp+38h] [rbp-148h]
  __int64 v42; // [rsp+40h] [rbp-140h]
  __int64 v43; // [rsp+48h] [rbp-138h]
  __int64 v44; // [rsp+50h] [rbp-130h]
  __int64 v45; // [rsp+58h] [rbp-128h]
  char v46; // [rsp+60h] [rbp-120h]
  __int64 v47; // [rsp+70h] [rbp-110h] BYREF
  _QWORD *v48; // [rsp+78h] [rbp-108h]
  __int64 v49; // [rsp+80h] [rbp-100h]
  __int64 v50; // [rsp+88h] [rbp-F8h]
  __int64 v51; // [rsp+90h] [rbp-F0h]
  __int64 v52; // [rsp+98h] [rbp-E8h]
  __int64 v53; // [rsp+A0h] [rbp-E0h]
  __int64 v54; // [rsp+A8h] [rbp-D8h]
  __int64 v55; // [rsp+B0h] [rbp-D0h]
  __int64 v56; // [rsp+B8h] [rbp-C8h]
  __int64 v57; // [rsp+C0h] [rbp-C0h]
  __int64 v58; // [rsp+C8h] [rbp-B8h]
  char v59; // [rsp+D0h] [rbp-B0h]
  __int64 v60; // [rsp+E0h] [rbp-A0h] BYREF
  void **v61; // [rsp+E8h] [rbp-98h]
  void **v62; // [rsp+F0h] [rbp-90h]
  int v63; // [rsp+F8h] [rbp-88h]
  int v64; // [rsp+FCh] [rbp-84h]
  __int64 v65; // [rsp+120h] [rbp-60h]
  unsigned __int64 v66; // [rsp+128h] [rbp-58h]
  int v67; // [rsp+134h] [rbp-4Ch]
  int v68; // [rsp+138h] [rbp-48h]

  v46 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v47 = 1;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  sub_185A140((__int64)&v47, 0);
  if ( !(_DWORD)v50 )
  {
    LODWORD(v49) = v49 + 1;
    BUG();
  }
  v4 = 1;
  v5 = 0;
  v6 = (v50 - 1) & (((unsigned int)&unk_4F9EE60 >> 9) ^ ((unsigned int)&unk_4F9EE60 >> 4));
  v7 = (void **)&v48[2 * v6];
  v8 = *v7;
  if ( *v7 != &unk_4F9EE60 )
  {
    while ( v8 != (void *)-8LL )
    {
      if ( v8 == (void *)-16LL && !v5 )
        v5 = v7;
      v6 = (v50 - 1) & (v4 + v6);
      v7 = (void **)&v48[2 * v6];
      v8 = *v7;
      if ( *v7 == &unk_4F9EE60 )
        goto LABEL_3;
      ++v4;
    }
    if ( v5 )
      v7 = v5;
  }
LABEL_3:
  LODWORD(v49) = v49 + 1;
  if ( *v7 != (void *)-8LL )
    --HIDWORD(v49);
  *v7 = &unk_4F9EE60;
  v7[1] = 0;
  v9 = (_QWORD *)sub_22077B0(16);
  if ( v9 )
  {
    *v9 = &unk_49F1470;
    v9[1] = &v34;
  }
  v10 = v7[1];
  v7[1] = v9;
  if ( v10 )
    (*(void (__fastcall **)(void *, void **))(*(_QWORD *)v10 + 8LL))(v10, v5);
  v11 = (__int64)a2;
  LODWORD(a2) = 1;
  sub_1858B90(&v60, a1 + 160, v11);
  if ( v67 == v68 )
  {
    v31 = v61;
    if ( v62 == v61 )
    {
      v32 = &v61[v64];
      if ( v61 == v32 )
      {
        v33 = v61;
      }
      else
      {
        do
        {
          if ( *v31 == &unk_4F9EE48 )
            break;
          ++v31;
        }
        while ( v32 != v31 );
        v33 = &v61[v64];
      }
    }
    else
    {
      a2 = &unk_4F9EE48;
      v32 = &v62[v63];
      v31 = (void **)sub_16CC9F0((__int64)&v60, (__int64)&unk_4F9EE48);
      if ( *v31 == &unk_4F9EE48 )
      {
        if ( v62 == v61 )
          v33 = &v62[v64];
        else
          v33 = &v62[v63];
      }
      else
      {
        if ( v62 != v61 )
        {
          v31 = &v62[v63];
LABEL_53:
          LOBYTE(a2) = v32 == v31;
          goto LABEL_10;
        }
        v31 = &v62[v64];
        v33 = v31;
      }
    }
    while ( v33 != v31 && (unsigned __int64)*v31 >= 0xFFFFFFFFFFFFFFFELL )
      ++v31;
    goto LABEL_53;
  }
LABEL_10:
  if ( v66 != v65 )
    _libc_free(v66);
  if ( v62 != v61 )
    _libc_free((unsigned __int64)v62);
  j___libc_free_0(v56);
  if ( (_DWORD)v54 )
  {
    v12 = (_QWORD **)(v52 + 32LL * (unsigned int)v54);
    for ( i = (_QWORD **)(v52 + 8); ; i += 4 )
    {
      v14 = (__int64)*(i - 1);
      if ( v14 != -8 && v14 != -16 )
      {
        v15 = *i;
        while ( v15 != i )
        {
          v16 = v15;
          v15 = (_QWORD *)*v15;
          v17 = v16[3];
          if ( v17 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
          j_j___libc_free_0(v16, 32);
        }
      }
      if ( v12 == i + 3 )
        break;
    }
  }
  j___libc_free_0(v52);
  if ( (_DWORD)v50 )
  {
    v18 = v48;
    v19 = &v48[2 * (unsigned int)v50];
    do
    {
      if ( *v18 != -16 && *v18 != -8 )
      {
        v20 = v18[1];
        if ( v20 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
      }
      v18 += 2;
    }
    while ( v19 != v18 );
  }
  j___libc_free_0(v48);
  j___libc_free_0(v43);
  if ( (_DWORD)v41 )
  {
    v21 = (_QWORD **)(v39 + 32LL * (unsigned int)v41);
    for ( j = (_QWORD **)(v39 + 8); ; j += 4 )
    {
      v23 = (__int64)*(j - 1);
      if ( v23 != -16 && v23 != -8 )
      {
        v24 = *j;
        while ( v24 != j )
        {
          v25 = v24;
          v24 = (_QWORD *)*v24;
          v26 = v25[3];
          if ( v26 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
          j_j___libc_free_0(v25, 32);
        }
      }
      if ( v21 == j + 3 )
        break;
    }
  }
  j___libc_free_0(v39);
  if ( (_DWORD)v37 )
  {
    v27 = v35;
    v28 = &v35[2 * (unsigned int)v37];
    do
    {
      if ( *v27 != -16 && *v27 != -8 )
      {
        v29 = v27[1];
        if ( v29 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v29 + 8LL))(v29);
      }
      v27 += 2;
    }
    while ( v28 != v27 );
  }
  j___libc_free_0(v35);
  return (unsigned int)a2;
}
