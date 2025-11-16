// Function: sub_888280
// Address: 0x888280
//
__int64 __fastcall sub_888280(__int64 a1, __int64 a2, int a3, int a4)
{
  int v6; // ebx
  unsigned int v7; // edx
  __int64 v8; // rax
  int v9; // r10d
  int v10; // r9d
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v14; // rax
  _QWORD *v15; // r14
  int v16; // esi
  unsigned int v17; // ebx
  __int64 v18; // rcx
  __int64 v19; // r9
  int *v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r13
  int v23; // eax
  unsigned int v24; // r12d
  unsigned int v25; // ebx
  _DWORD *v26; // rax
  __int64 v27; // r8
  __int64 *v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdi
  unsigned int v32; // r9d
  _QWORD *i; // rax
  unsigned int v34; // r13d
  int v35; // eax
  _DWORD *v36; // rax
  unsigned int v37; // r9d
  _QWORD *j; // rax
  __int64 v39; // [rsp+8h] [rbp-38h]

  v6 = a4 + 31 * (a3 + 527);
  v7 = v6 & qword_4F5FED8[1];
  v8 = *qword_4F5FED8 + 16LL * v7;
  v9 = *(_DWORD *)(v8 + 4);
  v10 = *(_DWORD *)v8;
  if ( a4 == v9 && a3 == v10 )
  {
LABEL_6:
    v11 = *(_QWORD *)(v8 + 8);
    if ( v11 )
    {
      if ( *(_QWORD *)(v11 + 8) != a1 || *(_QWORD *)(v11 + 16) != a2 )
      {
        v12 = sub_878940(a1, a2);
        *(_DWORD *)(v12 + 24) = a3;
        v11 = v12;
        *(_DWORD *)(v12 + 28) = a4;
      }
      return v11;
    }
  }
  else
  {
    while ( v10 | v9 )
    {
      v7 = qword_4F5FED8[1] & (v7 + 1);
      v8 = *qword_4F5FED8 + 16LL * v7;
      v10 = *(_DWORD *)v8;
      v9 = *(_DWORD *)(v8 + 4);
      if ( a3 == *(_DWORD *)v8 && a4 == v9 )
        goto LABEL_6;
    }
  }
  v14 = sub_878940(a1, a2);
  v15 = qword_4F5FED8;
  *(_DWORD *)(v14 + 24) = a3;
  v11 = v14;
  *(_DWORD *)(v14 + 28) = a4;
  v16 = *((_DWORD *)v15 + 2);
  v17 = v16 & v6;
  v18 = 16LL * v17;
  v19 = *v15 + v18;
  if ( *(_QWORD *)v19 )
  {
    do
    {
      v17 = v16 & (v17 + 1);
      v20 = (int *)(*v15 + 16LL * v17);
    }
    while ( *(_QWORD *)v20 );
    v39 = v18;
    sub_888210(v20, (__int64 *)(*v15 + v18));
    v21 = *v15 + v39;
    *(_DWORD *)v21 = a3;
    *(_DWORD *)(v21 + 4) = a4;
    if ( a4 | a3 )
      *(_QWORD *)(v21 + 8) = v11;
    v22 = *((unsigned int *)v15 + 2);
    v23 = *((_DWORD *)v15 + 3) + 1;
    *((_DWORD *)v15 + 3) = v23;
    if ( 2 * v23 > (unsigned int)v22 )
    {
      v24 = v22 + 1;
      v25 = 2 * v22 + 1;
      v26 = (_DWORD *)sub_823970(16LL * (unsigned int)(2 * v22 + 2));
      v29 = (unsigned int)(2 * v22 + 2);
      v30 = (__int64)v26;
      if ( 2 * (_DWORD)v22 != -2 )
      {
        v29 = (__int64)&v26[4 * v25 + 4];
        do
        {
          if ( v26 )
          {
            *v26 = 0;
            v26[1] = 0;
          }
          v26 += 4;
        }
        while ( (_DWORD *)v29 != v26 );
      }
      v31 = *v15;
      if ( (_DWORD)v22 != -1 )
      {
        v29 = *v15;
        do
        {
          v28 = (__int64 *)*(unsigned int *)(v29 + 4);
          if ( *(_QWORD *)v29 )
          {
            v32 = v25 & ((_DWORD)v28 + 31 * (*(_DWORD *)v29 + 527));
            for ( i = (_QWORD *)(v30 + 16LL * v32); *i; i = (_QWORD *)(v30 + 16LL * v32) )
              v32 = v25 & (v32 + 1);
            v27 = *(_QWORD *)v29;
            *i = *(_QWORD *)v29;
            v28 = (__int64 *)((unsigned int)v27 | HIDWORD(v27));
            if ( v27 )
            {
              v27 = *(_QWORD *)(v29 + 8);
              i[1] = v27;
            }
          }
          v29 += 16;
        }
        while ( v31 + 16 * v22 + 16 != v29 );
      }
LABEL_30:
      *v15 = v30;
      *((_DWORD *)v15 + 2) = v25;
      sub_823A00(v31, 16LL * v24, v29, v30, v27, v28);
    }
  }
  else
  {
    *(_DWORD *)v19 = a3;
    *(_DWORD *)(v19 + 4) = a4;
    if ( a4 | a3 )
      *(_QWORD *)(v19 + 8) = v14;
    v34 = *((_DWORD *)v15 + 2);
    v35 = *((_DWORD *)v15 + 3) + 1;
    *((_DWORD *)v15 + 3) = v35;
    if ( 2 * v35 > v34 )
    {
      v24 = v34 + 1;
      v25 = 2 * v34 + 1;
      v36 = (_DWORD *)sub_823970(16LL * (2 * v34 + 2));
      v29 = 2 * v34 + 2;
      v30 = (__int64)v36;
      if ( 2 * v34 != -2 )
      {
        v29 = (__int64)&v36[4 * v25 + 4];
        do
        {
          if ( v36 )
          {
            *v36 = 0;
            v36[1] = 0;
          }
          v36 += 4;
        }
        while ( (_DWORD *)v29 != v36 );
      }
      v31 = *v15;
      if ( v34 != -1 )
      {
        v29 = *v15;
        do
        {
          v28 = (__int64 *)*(unsigned int *)(v29 + 4);
          if ( *(_QWORD *)v29 )
          {
            v37 = v25 & ((_DWORD)v28 + 31 * (*(_DWORD *)v29 + 527));
            for ( j = (_QWORD *)(v30 + 16LL * v37); *j; j = (_QWORD *)(v30 + 16LL * v37) )
              v37 = v25 & (v37 + 1);
            v27 = *(_QWORD *)v29;
            *j = *(_QWORD *)v29;
            v28 = (__int64 *)((unsigned int)v27 | HIDWORD(v27));
            if ( v27 )
            {
              v27 = *(_QWORD *)(v29 + 8);
              j[1] = v27;
            }
          }
          v29 += 16;
        }
        while ( v31 + 16LL * v34 + 16 != v29 );
      }
      goto LABEL_30;
    }
  }
  return v11;
}
