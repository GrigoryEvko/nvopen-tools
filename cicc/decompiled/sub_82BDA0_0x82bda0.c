// Function: sub_82BDA0
// Address: 0x82bda0
//
__int64 __fastcall sub_82BDA0(unsigned __int64 a1)
{
  unsigned __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rbx
  int v6; // edi
  __int64 v7; // rsi
  unsigned int v8; // ecx
  unsigned int v9; // edx
  __int64 v10; // r8
  __int64 v11; // r10
  __int64 v12; // r9
  __int64 v13; // r11
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // r8
  __int64 v17; // r14
  int v18; // eax
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // r14
  int v22; // eax
  unsigned int v23; // r13d
  unsigned int v24; // r12d
  _QWORD *v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // r9
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rsi
  unsigned __int64 v31; // rdi
  unsigned __int64 j; // rdx
  unsigned int v33; // edx
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rsi
  unsigned __int64 v37; // rdi
  unsigned __int64 i; // rdx
  unsigned int v39; // edx
  __int64 v40; // rax
  _QWORD *v41; // rax
  _QWORD *v42; // rcx
  _QWORD *v43; // rdx

  v2 = a1 >> 3;
  v3 = sub_82BD70();
  v4 = *(_QWORD *)(v3 + 1008) + 8 * (5LL * *(_QWORD *)(v3 + 1024) - 5);
  v5 = *(_QWORD *)(v4 + 32);
  if ( !v5 )
  {
    v5 = sub_823970(16);
    if ( v5 )
    {
      v41 = (_QWORD *)sub_823970(128);
      v42 = v41;
      v43 = v41 + 16;
      do
      {
        if ( v41 )
          *v41 = 0;
        v41 += 2;
      }
      while ( v43 != v41 );
      *(_QWORD *)v5 = v42;
      *(_QWORD *)(v5 + 8) = 7;
    }
    *(_QWORD *)(v4 + 32) = v5;
    v6 = *(_DWORD *)(v5 + 8);
    v7 = *(_QWORD *)v5;
    v8 = v6 & v2;
    v10 = 16LL * (v6 & (unsigned int)v2);
    v11 = *(_QWORD *)v5 + v10;
    v12 = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 )
      goto LABEL_7;
    goto LABEL_14;
  }
  v6 = *(_DWORD *)(v5 + 8);
  v7 = *(_QWORD *)v5;
  v8 = v6 & v2;
  v9 = v6 & v2;
  v10 = 16LL * (v6 & (unsigned int)v2);
  v11 = *(_QWORD *)v5 + v10;
  v12 = *(_QWORD *)v11;
  v13 = *(_QWORD *)v11;
  if ( a1 != *(_QWORD *)v11 )
  {
    while ( v13 )
    {
      v9 = v6 & (v9 + 1);
      v20 = v7 + 16LL * v9;
      v13 = *(_QWORD *)v20;
      if ( a1 == *(_QWORD *)v20 )
        goto LABEL_12;
    }
LABEL_6:
    if ( v12 )
    {
      do
      {
LABEL_7:
        v8 = v6 & (v8 + 1);
        v14 = v7 + 16LL * v8;
      }
      while ( *(_QWORD *)v14 );
      v15 = *(_DWORD *)(v11 + 8);
      *(_QWORD *)v14 = v12;
      *(_DWORD *)(v14 + 8) = v15;
      *(_QWORD *)v11 = 0;
      v16 = *(_QWORD *)v5 + v10;
      *(_QWORD *)v16 = a1;
      if ( a1 )
        *(_DWORD *)(v16 + 8) = 1;
      v17 = *(unsigned int *)(v5 + 8);
      v18 = *(_DWORD *)(v5 + 12) + 1;
      *(_DWORD *)(v5 + 12) = v18;
      if ( 2 * v18 <= (unsigned int)v17 )
        return 0;
      v23 = v17 + 1;
      v24 = 2 * v17 + 1;
      v35 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v17 + 2));
      v28 = (__int64)v35;
      if ( 2 * (_DWORD)v17 != -2 )
      {
        v26 = (__int64)&v35[2 * v24 + 2];
        do
        {
          if ( v35 )
            *v35 = 0;
          v35 += 2;
        }
        while ( (_QWORD *)v26 != v35 );
      }
      v29 = *(_QWORD *)v5;
      if ( (_DWORD)v17 != -1 )
      {
        v36 = *(_QWORD *)v5;
        v27 = (__int64 *)(v29 + 16 * v17 + 16);
        do
        {
          while ( 1 )
          {
            v37 = *(_QWORD *)v36;
            if ( *(_QWORD *)v36 )
              break;
            v36 += 16;
            if ( v27 == (__int64 *)v36 )
              goto LABEL_43;
          }
          for ( i = v37 >> 3; ; LODWORD(i) = v39 + 1 )
          {
            v39 = v24 & i;
            v40 = v28 + 16LL * v39;
            if ( !*(_QWORD *)v40 )
              break;
          }
          *(_QWORD *)v40 = v37;
          v26 = *(unsigned int *)(v36 + 8);
          v36 += 16;
          *(_DWORD *)(v40 + 8) = v26;
        }
        while ( v27 != (__int64 *)v36 );
      }
LABEL_43:
      *(_QWORD *)v5 = v28;
      *(_DWORD *)(v5 + 8) = v24;
      sub_823A00(v29, 16LL * v23, v26, v28, v29, v27);
      return 0;
    }
LABEL_14:
    *(_QWORD *)v11 = a1;
    if ( a1 )
      *(_DWORD *)(v11 + 8) = 1;
    v21 = *(unsigned int *)(v5 + 8);
    v22 = *(_DWORD *)(v5 + 12) + 1;
    *(_DWORD *)(v5 + 12) = v22;
    if ( 2 * v22 <= (unsigned int)v21 )
      return 0;
    v23 = v21 + 1;
    v24 = 2 * v21 + 1;
    v25 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v21 + 2));
    v28 = (__int64)v25;
    if ( 2 * (_DWORD)v21 != -2 )
    {
      v26 = (__int64)&v25[2 * v24 + 2];
      do
      {
        if ( v25 )
          *v25 = 0;
        v25 += 2;
      }
      while ( (_QWORD *)v26 != v25 );
    }
    v29 = *(_QWORD *)v5;
    if ( (_DWORD)v21 != -1 )
    {
      v30 = *(_QWORD *)v5;
      v27 = (__int64 *)(v29 + 16 * v21 + 16);
      do
      {
        v31 = *(_QWORD *)v30;
        if ( *(_QWORD *)v30 )
        {
          for ( j = v31 >> 3; ; LODWORD(j) = v33 + 1 )
          {
            v33 = v24 & j;
            v34 = v28 + 16LL * v33;
            if ( !*(_QWORD *)v34 )
              break;
          }
          *(_QWORD *)v34 = v31;
          v26 = *(unsigned int *)(v30 + 8);
          *(_DWORD *)(v34 + 8) = v26;
        }
        v30 += 16;
      }
      while ( v27 != (__int64 *)v30 );
    }
    goto LABEL_43;
  }
  v20 = *(_QWORD *)v5 + v10;
LABEL_12:
  result = *(unsigned int *)(v20 + 8);
  if ( !(_DWORD)result )
    goto LABEL_6;
  return result;
}
