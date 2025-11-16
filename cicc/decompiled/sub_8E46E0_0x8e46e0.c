// Function: sub_8E46E0
// Address: 0x8e46e0
//
void __fastcall sub_8E46E0(__int64 a1, __int64 a2, __int64 *a3, int a4)
{
  int v7; // esi
  __int64 v8; // rdx
  unsigned int v9; // ecx
  __int64 v10; // rdi
  _QWORD *v11; // r10
  __int64 v12; // r11
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r14
  int v18; // eax
  unsigned int v19; // r13d
  unsigned int v20; // r12d
  _QWORD *v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // r9
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 *v26; // rsi
  unsigned __int64 v27; // rdi
  unsigned __int64 i; // rdx
  unsigned int v29; // edx
  unsigned __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // r14
  int v33; // eax
  _QWORD *v34; // rax
  __int64 *v35; // rsi
  unsigned __int64 v36; // rdi
  unsigned __int64 j; // rdx
  unsigned int v38; // edx
  unsigned __int64 *v39; // rax

  v7 = *(_DWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v9 = v7 & a4;
  v10 = 16LL * v9;
  v11 = (_QWORD *)(v8 + v10);
  v12 = *(_QWORD *)(v8 + v10);
  if ( v12 )
  {
    do
    {
      v9 = v7 & (v9 + 1);
      v13 = (_QWORD *)(v8 + 16LL * v9);
    }
    while ( *v13 );
    v14 = v11[1];
    *v13 = v12;
    v13[1] = v14;
    *v11 = 0;
    v15 = (_QWORD *)(*(_QWORD *)a1 + v10);
    v16 = *a3;
    *v15 = a2;
    if ( a2 )
      v15[1] = v16;
    v17 = *(unsigned int *)(a1 + 8);
    v18 = *(_DWORD *)(a1 + 12) + 1;
    *(_DWORD *)(a1 + 12) = v18;
    if ( 2 * v18 <= (unsigned int)v17 )
      return;
    v19 = v17 + 1;
    v20 = 2 * v17 + 1;
    v21 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v17 + 2));
    v24 = (__int64)v21;
    if ( 2 * (_DWORD)v17 != -2 )
    {
      v22 = (__int64)&v21[2 * v20 + 2];
      do
      {
        if ( v21 )
          *v21 = 0;
        v21 += 2;
      }
      while ( (_QWORD *)v22 != v21 );
    }
    v25 = *(_QWORD *)a1;
    if ( (_DWORD)v17 != -1 )
    {
      v26 = *(__int64 **)a1;
      v23 = (__int64 *)(v25 + 16 * v17 + 16);
      do
      {
        while ( 1 )
        {
          v27 = *v26;
          if ( *v26 )
            break;
          v26 += 2;
          if ( v23 == v26 )
            goto LABEL_20;
        }
        for ( i = v27 >> 3; ; LODWORD(i) = v29 + 1 )
        {
          v29 = v20 & i;
          v30 = (unsigned __int64 *)(v24 + 16LL * v29);
          if ( !*v30 )
            break;
        }
        *v30 = v27;
        v22 = v26[1];
        v26 += 2;
        v30[1] = v22;
      }
      while ( v23 != v26 );
    }
LABEL_20:
    *(_DWORD *)(a1 + 8) = v20;
    *(_QWORD *)a1 = v24;
    sub_823A00(v25, 16LL * v19, v22, v24, v25, v23);
    return;
  }
  v31 = *a3;
  *v11 = a2;
  if ( a2 )
    v11[1] = v31;
  v32 = *(unsigned int *)(a1 + 8);
  v33 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v33;
  if ( 2 * v33 > (unsigned int)v32 )
  {
    v19 = v32 + 1;
    v20 = 2 * v32 + 1;
    v34 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v32 + 2));
    v24 = (__int64)v34;
    if ( 2 * (_DWORD)v32 != -2 )
    {
      v22 = (__int64)&v34[2 * v20 + 2];
      do
      {
        if ( v34 )
          *v34 = 0;
        v34 += 2;
      }
      while ( (_QWORD *)v22 != v34 );
    }
    v25 = *(_QWORD *)a1;
    if ( (_DWORD)v32 != -1 )
    {
      v35 = *(__int64 **)a1;
      v23 = (__int64 *)(v25 + 16 * v32 + 16);
      do
      {
        v36 = *v35;
        if ( *v35 )
        {
          for ( j = v36 >> 3; ; LODWORD(j) = v38 + 1 )
          {
            v38 = v20 & j;
            v39 = (unsigned __int64 *)(v24 + 16LL * v38);
            if ( !*v39 )
              break;
          }
          *v39 = v36;
          v22 = v35[1];
          v39[1] = v22;
        }
        v35 += 2;
      }
      while ( v35 != v23 );
    }
    goto LABEL_20;
  }
}
