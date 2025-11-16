// Function: sub_603C70
// Address: 0x603c70
//
__int64 __fastcall sub_603C70(__int64 *a1, int a2, __int64 *a3, int a4)
{
  int v7; // esi
  int *v8; // rdx
  unsigned int v9; // ecx
  __int64 v10; // rdi
  int *v11; // r10
  int v12; // r11d
  __int64 v13; // rax
  __int64 v14; // r14
  int v15; // eax
  __int64 result; // rax
  int *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // r14
  int v22; // eax
  unsigned int v23; // r13d
  int v24; // r12d
  _DWORD *v25; // rax
  _DWORD *v26; // rsi
  _DWORD *v27; // rdx
  int *v28; // r8
  int *v29; // rcx
  __int64 v30; // r9
  int v31; // edi
  unsigned int j; // edx
  _DWORD *v33; // rax
  __int64 v34; // rdx
  _DWORD *v35; // rax
  _DWORD *v36; // rdx
  int *v37; // rcx
  int v38; // edi
  unsigned int i; // edx
  _DWORD *v40; // rax

  v7 = *((_DWORD *)a1 + 2);
  v8 = (int *)*a1;
  v9 = v7 & a4;
  v10 = 4LL * v9;
  v11 = &v8[v10];
  v12 = v8[v10];
  if ( !v12 )
  {
    v13 = *a3;
    *v11 = a2;
    if ( a2 )
      *((_QWORD *)v11 + 1) = v13;
    v14 = *((unsigned int *)a1 + 2);
    v15 = *((_DWORD *)a1 + 3) + 1;
    *((_DWORD *)a1 + 3) = v15;
    result = (unsigned int)(2 * v15);
    if ( (unsigned int)result <= (unsigned int)v14 )
      return result;
    v23 = v14 + 1;
    v24 = 2 * v14 + 1;
    v35 = (_DWORD *)sub_823970(16LL * (unsigned int)(2 * v14 + 2));
    v26 = v35;
    if ( 2 * (_DWORD)v14 != -2 )
    {
      v36 = &v35[4 * v24 + 4];
      do
      {
        if ( v35 )
          *v35 = 0;
        v35 += 4;
      }
      while ( v36 != v35 );
    }
    v28 = (int *)*a1;
    if ( (_DWORD)v14 != -1 )
    {
      v37 = (int *)*a1;
      do
      {
        v38 = *v37;
        if ( *v37 )
        {
          for ( i = v38 & v24; ; i = v24 & (i + 1) )
          {
            v40 = &v26[4 * i];
            if ( !*v40 )
              break;
          }
          *v40 = v38;
          *((_QWORD *)v40 + 1) = *((_QWORD *)v37 + 1);
        }
        v37 += 4;
      }
      while ( v37 != &v28[4 * v14 + 4] );
    }
LABEL_23:
    *a1 = (__int64)v26;
    *((_DWORD *)a1 + 2) = v24;
    return sub_823A00(v28, 16LL * v23);
  }
  do
  {
    v9 = v7 & (v9 + 1);
    v17 = &v8[4 * v9];
  }
  while ( *v17 );
  v18 = *((_QWORD *)v11 + 1);
  *v17 = v12;
  *((_QWORD *)v17 + 1) = v18;
  *v11 = 0;
  v19 = *a1 + v10 * 4;
  v20 = *a3;
  *(_DWORD *)v19 = a2;
  if ( a2 )
    *(_QWORD *)(v19 + 8) = v20;
  v21 = *((unsigned int *)a1 + 2);
  v22 = *((_DWORD *)a1 + 3) + 1;
  *((_DWORD *)a1 + 3) = v22;
  result = (unsigned int)(2 * v22);
  if ( (unsigned int)result > (unsigned int)v21 )
  {
    v23 = v21 + 1;
    v24 = 2 * v21 + 1;
    v25 = (_DWORD *)sub_823970(16LL * (unsigned int)(2 * v21 + 2));
    v26 = v25;
    if ( 2 * (_DWORD)v21 != -2 )
    {
      v27 = &v25[4 * v24 + 4];
      do
      {
        if ( v25 )
          *v25 = 0;
        v25 += 4;
      }
      while ( v27 != v25 );
    }
    v28 = (int *)*a1;
    if ( (_DWORD)v21 != -1 )
    {
      v29 = (int *)*a1;
      v30 = (__int64)&v28[4 * v21 + 4];
      do
      {
        while ( 1 )
        {
          v31 = *v29;
          if ( *v29 )
            break;
          v29 += 4;
          if ( (int *)v30 == v29 )
            goto LABEL_23;
        }
        for ( j = v31 & v24; ; j = v24 & (j + 1) )
        {
          v33 = &v26[4 * j];
          if ( !*v33 )
            break;
        }
        *v33 = v31;
        v34 = *((_QWORD *)v29 + 1);
        v29 += 4;
        *((_QWORD *)v33 + 1) = v34;
      }
      while ( (int *)v30 != v29 );
    }
    goto LABEL_23;
  }
  return result;
}
