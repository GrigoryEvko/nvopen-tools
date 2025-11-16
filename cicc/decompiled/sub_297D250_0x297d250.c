// Function: sub_297D250
// Address: 0x297d250
//
void __fastcall sub_297D250(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // r13
  _QWORD *v6; // r14
  unsigned __int64 v7; // rdi
  __int64 v8; // r14
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rbx
  _QWORD *v22; // r15
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rbx
  _QWORD *v30; // r15
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // rbx
  _QWORD *v38; // r15
  unsigned __int64 v39; // rdi
  __int64 v40; // rdi
  _QWORD **v41; // r12
  _QWORD *v42; // rbx
  unsigned __int64 v43; // rdi

  v2 = *(_QWORD *)(a1 + 312);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 288);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(unsigned int *)(a1 + 280);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD **)(a1 + 264);
    v6 = &v5[6 * v4];
    do
    {
      if ( *v5 != -4096 && *v5 != -8192 )
      {
        v7 = v5[1];
        if ( (_QWORD *)v7 != v5 + 3 )
          _libc_free(v7);
      }
      v5 += 6;
    }
    while ( v6 != v5 );
    v4 = *(unsigned int *)(a1 + 280);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 264), 48 * v4, 8);
  v8 = *(_QWORD *)(a1 + 240);
  v9 = v8 + 32LL * *(unsigned int *)(a1 + 248);
  if ( v8 != v9 )
  {
    do
    {
      v10 = *(_QWORD *)(v9 - 24);
      v9 -= 32LL;
      if ( v10 )
        j_j___libc_free_0(v10);
    }
    while ( v8 != v9 );
    v9 = *(_QWORD *)(a1 + 240);
  }
  if ( a1 + 256 != v9 )
    _libc_free(v9);
  sub_C7D6A0(*(_QWORD *)(a1 + 216), 16LL * *(unsigned int *)(a1 + 232), 8);
  v11 = *(unsigned int *)(a1 + 200);
  if ( (_DWORD)v11 )
  {
    v12 = *(_QWORD *)(a1 + 184);
    v13 = v12 + 72 * v11;
    do
    {
      if ( *(_QWORD *)v12 != -4096 && *(_QWORD *)v12 != -8192 )
      {
        v14 = *(_QWORD *)(v12 + 40);
        if ( v14 != v12 + 56 )
          _libc_free(v14);
        sub_C7D6A0(*(_QWORD *)(v12 + 16), 8LL * *(unsigned int *)(v12 + 32), 8);
      }
      v12 += 72;
    }
    while ( v13 != v12 );
    v11 = *(unsigned int *)(a1 + 200);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 184), 72 * v11, 8);
  v15 = *(_QWORD *)(a1 + 152);
  if ( v15 )
    j_j___libc_free_0(v15);
  v16 = *(unsigned int *)(a1 + 144);
  if ( (_DWORD)v16 )
  {
    v17 = *(_QWORD *)(a1 + 128);
    v18 = v17 + 56 * v16;
    while ( 1 )
    {
      v19 = *(_QWORD *)(v17 + 16);
      if ( v19 == -4096 )
      {
        if ( *(_QWORD *)(v17 + 8) != -4096 || *(_QWORD *)v17 != -4096 )
        {
LABEL_35:
          v20 = *(unsigned int *)(v17 + 48);
          if ( (_DWORD)v20 )
          {
            v21 = *(_QWORD **)(v17 + 32);
            v22 = &v21[11 * v20];
            do
            {
              if ( *v21 != -4096 && *v21 != -8192 )
              {
                v23 = v21[1];
                if ( (_QWORD *)v23 != v21 + 3 )
                  _libc_free(v23);
              }
              v21 += 11;
            }
            while ( v22 != v21 );
            v20 = *(unsigned int *)(v17 + 48);
          }
          sub_C7D6A0(*(_QWORD *)(v17 + 32), 88 * v20, 8);
        }
      }
      else if ( v19 != -8192 || *(_QWORD *)(v17 + 8) != -8192 || *(_QWORD *)v17 != -8192 )
      {
        goto LABEL_35;
      }
      v17 += 56;
      if ( v18 == v17 )
      {
        v16 = *(unsigned int *)(a1 + 144);
        break;
      }
    }
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 56 * v16, 8);
  v24 = *(unsigned int *)(a1 + 112);
  if ( !(_DWORD)v24 )
    goto LABEL_61;
  v25 = *(_QWORD *)(a1 + 96);
  v26 = v25 + 56 * v24;
  do
  {
    v27 = *(_QWORD *)(v25 + 16);
    if ( v27 == -4096 )
    {
      if ( *(_QWORD *)(v25 + 8) == -4096 && *(_QWORD *)v25 == -4096 )
        goto LABEL_59;
    }
    else if ( v27 == -8192 && *(_QWORD *)(v25 + 8) == -8192 && *(_QWORD *)v25 == -8192 )
    {
      goto LABEL_59;
    }
    v28 = *(unsigned int *)(v25 + 48);
    if ( (_DWORD)v28 )
    {
      v29 = *(_QWORD **)(v25 + 32);
      v30 = &v29[11 * v28];
      do
      {
        if ( *v29 != -8192 && *v29 != -4096 )
        {
          v31 = v29[1];
          if ( (_QWORD *)v31 != v29 + 3 )
            _libc_free(v31);
        }
        v29 += 11;
      }
      while ( v30 != v29 );
      v28 = *(unsigned int *)(v25 + 48);
    }
    sub_C7D6A0(*(_QWORD *)(v25 + 32), 88 * v28, 8);
LABEL_59:
    v25 += 56;
  }
  while ( v26 != v25 );
  v24 = *(unsigned int *)(a1 + 112);
LABEL_61:
  sub_C7D6A0(*(_QWORD *)(a1 + 96), 56 * v24, 8);
  v32 = *(unsigned int *)(a1 + 80);
  if ( !(_DWORD)v32 )
    goto LABEL_76;
  v33 = *(_QWORD *)(a1 + 64);
  v34 = v33 + 56 * v32;
  while ( 2 )
  {
    v35 = *(_QWORD *)(v33 + 16);
    if ( v35 == -4096 )
    {
      if ( *(_QWORD *)(v33 + 8) != -4096 || *(_QWORD *)v33 != -4096 )
        goto LABEL_65;
    }
    else
    {
      if ( v35 == -8192 && *(_QWORD *)(v33 + 8) == -8192 && *(_QWORD *)v33 == -8192 )
        goto LABEL_74;
LABEL_65:
      v36 = *(unsigned int *)(v33 + 48);
      if ( (_DWORD)v36 )
      {
        v37 = *(_QWORD **)(v33 + 32);
        v38 = &v37[11 * v36];
        do
        {
          if ( *v37 != -8192 && *v37 != -4096 )
          {
            v39 = v37[1];
            if ( (_QWORD *)v39 != v37 + 3 )
              _libc_free(v39);
          }
          v37 += 11;
        }
        while ( v38 != v37 );
        v36 = *(unsigned int *)(v33 + 48);
      }
      sub_C7D6A0(*(_QWORD *)(v33 + 32), 88 * v36, 8);
    }
LABEL_74:
    v33 += 56;
    if ( v34 != v33 )
      continue;
    break;
  }
  v32 = *(unsigned int *)(a1 + 80);
LABEL_76:
  v40 = *(_QWORD *)(a1 + 64);
  v41 = (_QWORD **)(a1 + 32);
  sub_C7D6A0(v40, 56 * v32, 8);
  v42 = *v41;
  while ( v42 != v41 )
  {
    v43 = (unsigned __int64)v42;
    v42 = (_QWORD *)*v42;
    j_j___libc_free_0(v43);
  }
}
