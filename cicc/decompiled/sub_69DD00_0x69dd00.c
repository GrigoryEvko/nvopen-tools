// Function: sub_69DD00
// Address: 0x69dd00
//
__int64 __fastcall sub_69DD00(__int64 *a1, __int64 a2, int *a3, int a4)
{
  int v7; // esi
  __int64 *v8; // rdx
  unsigned int v9; // ecx
  __int64 v10; // rdi
  __int64 *v11; // r10
  __int64 v12; // r11
  __int64 *v13; // rax
  int v14; // edx
  __int64 v15; // rdi
  int v16; // eax
  __int64 v17; // r14
  int v18; // eax
  __int64 result; // rax
  unsigned int v20; // r13d
  unsigned int v21; // r12d
  _QWORD *v22; // rax
  _QWORD *v23; // rcx
  _QWORD *v24; // rdx
  __int64 *v25; // r8
  unsigned __int64 *v26; // rsi
  __int64 v27; // r9
  unsigned __int64 v28; // rdi
  unsigned __int64 i; // rdx
  unsigned int v30; // edx
  unsigned __int64 *v31; // rax
  int v32; // edx
  int v33; // eax
  __int64 v34; // r14
  int v35; // eax
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  unsigned __int64 *v38; // rsi
  unsigned __int64 v39; // rdi
  unsigned __int64 j; // rdx
  unsigned int v41; // edx
  unsigned __int64 *v42; // rax

  v7 = *((_DWORD *)a1 + 2);
  v8 = (__int64 *)*a1;
  v9 = v7 & a4;
  v10 = 2LL * v9;
  v11 = &v8[v10];
  v12 = v8[v10];
  if ( v12 )
  {
    do
    {
      v9 = v7 & (v9 + 1);
      v13 = &v8[2 * v9];
    }
    while ( *v13 );
    v14 = *((_DWORD *)v11 + 2);
    *v13 = v12;
    *((_DWORD *)v13 + 2) = v14;
    *v11 = 0;
    v15 = *a1 + v10 * 8;
    v16 = *a3;
    *(_QWORD *)v15 = a2;
    if ( a2 )
      *(_DWORD *)(v15 + 8) = v16;
    v17 = *((unsigned int *)a1 + 2);
    v18 = *((_DWORD *)a1 + 3) + 1;
    *((_DWORD *)a1 + 3) = v18;
    result = (unsigned int)(2 * v18);
    if ( (unsigned int)result <= (unsigned int)v17 )
      return result;
    v20 = v17 + 1;
    v21 = 2 * v17 + 1;
    v22 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v17 + 2));
    v23 = v22;
    if ( 2 * (_DWORD)v17 != -2 )
    {
      v24 = &v22[2 * v21 + 2];
      do
      {
        if ( v22 )
          *v22 = 0;
        v22 += 2;
      }
      while ( v24 != v22 );
    }
    v25 = (__int64 *)*a1;
    if ( (_DWORD)v17 != -1 )
    {
      v26 = (unsigned __int64 *)*a1;
      v27 = (__int64)&v25[2 * v17 + 2];
      do
      {
        while ( 1 )
        {
          v28 = *v26;
          if ( *v26 )
            break;
          v26 += 2;
          if ( (unsigned __int64 *)v27 == v26 )
            goto LABEL_20;
        }
        for ( i = v28 >> 3; ; LODWORD(i) = v30 + 1 )
        {
          v30 = v21 & i;
          v31 = &v23[2 * v30];
          if ( !*v31 )
            break;
        }
        *v31 = v28;
        v32 = *((_DWORD *)v26 + 2);
        v26 += 2;
        *((_DWORD *)v31 + 2) = v32;
      }
      while ( (unsigned __int64 *)v27 != v26 );
    }
LABEL_20:
    *((_DWORD *)a1 + 2) = v21;
    *a1 = (__int64)v23;
    return sub_823A00(v25, 16LL * v20);
  }
  v33 = *a3;
  *v11 = a2;
  if ( a2 )
    *((_DWORD *)v11 + 2) = v33;
  v34 = *((unsigned int *)a1 + 2);
  v35 = *((_DWORD *)a1 + 3) + 1;
  *((_DWORD *)a1 + 3) = v35;
  result = (unsigned int)(2 * v35);
  if ( (unsigned int)result > (unsigned int)v34 )
  {
    v20 = v34 + 1;
    v21 = 2 * v34 + 1;
    v36 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v34 + 2));
    v23 = v36;
    if ( 2 * (_DWORD)v34 != -2 )
    {
      v37 = &v36[2 * v21 + 2];
      do
      {
        if ( v36 )
          *v36 = 0;
        v36 += 2;
      }
      while ( v37 != v36 );
    }
    v25 = (__int64 *)*a1;
    if ( (_DWORD)v34 != -1 )
    {
      v38 = (unsigned __int64 *)*a1;
      do
      {
        v39 = *v38;
        if ( *v38 )
        {
          for ( j = v39 >> 3; ; LODWORD(j) = v41 + 1 )
          {
            v41 = v21 & j;
            v42 = &v23[2 * v41];
            if ( !*v42 )
              break;
          }
          *v42 = v39;
          *((_DWORD *)v42 + 2) = *((_DWORD *)v38 + 2);
        }
        v38 += 2;
      }
      while ( v38 != (unsigned __int64 *)&v25[2 * v34 + 2] );
    }
    goto LABEL_20;
  }
  return result;
}
