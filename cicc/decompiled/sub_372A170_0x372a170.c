// Function: sub_372A170
// Address: 0x372a170
//
void __fastcall sub_372A170(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rsi
  __int64 v8; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  char *v13; // rax
  __int64 v14; // rcx
  char *v15; // r10
  __int64 v16; // r11
  char *v17; // rdx
  char *v18; // rax
  __int64 v19; // r10
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // ecx
  int *v23; // rax
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // r12

  v5 = a1 + 1;
  v8 = a1[2];
  v10 = a1 + 1;
  if ( v8 )
  {
    do
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(v8 + 16);
        v12 = *(_QWORD *)(v8 + 24);
        if ( *(_DWORD *)(v8 + 32) >= a2 )
          break;
        v8 = *(_QWORD *)(v8 + 24);
        if ( !v12 )
          goto LABEL_6;
      }
      v10 = (_QWORD *)v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
    while ( v11 );
LABEL_6:
    if ( v5 != v10 && *((_DWORD *)v10 + 8) > a2 )
      v10 = v5;
  }
  v13 = (char *)v10[5];
  v14 = *((unsigned int *)v10 + 12);
  v15 = &v13[16 * v14];
  v16 = (16 * v14) >> 4;
  if ( (16 * v14) >> 6 )
  {
    v17 = &v13[64 * ((16 * v14) >> 6)];
    while ( 1 )
    {
      if ( *(_QWORD *)v13 == a3 )
      {
        if ( *((_QWORD *)v13 + 1) == a4 )
          goto LABEL_21;
        if ( *((_QWORD *)v13 + 2) != a3 )
          goto LABEL_13;
      }
      else if ( *((_QWORD *)v13 + 2) != a3 )
      {
        goto LABEL_13;
      }
      if ( *((_QWORD *)v13 + 3) == a4 )
      {
        v13 += 16;
        goto LABEL_21;
      }
LABEL_13:
      if ( *((_QWORD *)v13 + 4) == a3 && *((_QWORD *)v13 + 5) == a4 )
      {
        v13 += 32;
        goto LABEL_21;
      }
      if ( *((_QWORD *)v13 + 6) == a3 && *((_QWORD *)v13 + 7) == a4 )
      {
        v13 += 48;
        goto LABEL_21;
      }
      v13 += 64;
      if ( v17 == v13 )
      {
        v16 = (v15 - v13) >> 4;
        break;
      }
    }
  }
  if ( v16 == 2 )
  {
LABEL_39:
    if ( *(_QWORD *)v13 == a3 && *((_QWORD *)v13 + 1) == a4 )
      goto LABEL_21;
    v13 += 16;
    goto LABEL_41;
  }
  if ( v16 == 3 )
  {
    if ( *(_QWORD *)v13 == a3 && *((_QWORD *)v13 + 1) == a4 )
      goto LABEL_21;
    v13 += 16;
    goto LABEL_39;
  }
  if ( v16 != 1 )
  {
LABEL_20:
    v13 = v15;
    goto LABEL_21;
  }
LABEL_41:
  if ( *(_QWORD *)v13 != a3 )
    goto LABEL_20;
  if ( *((_QWORD *)v13 + 1) != a4 )
    v13 = v15;
LABEL_21:
  v18 = v13 + 16;
  v19 = v15 - v18;
  v20 = v19 >> 4;
  if ( v19 > 0 )
  {
    do
    {
      v21 = *(_QWORD *)v18;
      v18 += 16;
      *((_QWORD *)v18 - 4) = v21;
      *((_QWORD *)v18 - 3) = *((_QWORD *)v18 - 1);
      --v20;
    }
    while ( v20 );
    LODWORD(v14) = *((_DWORD *)v10 + 12);
  }
  v22 = v14 - 1;
  *((_DWORD *)v10 + 12) = v22;
  if ( !v22 )
  {
    v23 = sub_220F330((int *)v10, v5);
    v24 = *((_QWORD *)v23 + 5);
    v25 = (unsigned __int64)v23;
    if ( (int *)v24 != v23 + 14 )
      _libc_free(v24);
    j_j___libc_free_0(v25);
    --a1[5];
  }
}
