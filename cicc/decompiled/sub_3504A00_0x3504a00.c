// Function: sub_3504A00
// Address: 0x3504a00
//
void __fastcall sub_3504A00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 *v8; // r13
  __int64 v9; // r15
  int v10; // esi
  __int64 v11; // rdx
  __int64 v12; // rdi
  int v13; // esi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // r13
  int v26; // eax
  __int64 v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v7 = 0;
  v8 = *(__int64 **)a2;
  v9 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( v9 != *(_QWORD *)a2 )
  {
    v10 = *(_DWORD *)(a3 + 24);
    v11 = *v8;
    v12 = *(_QWORD *)(a3 + 8);
    if ( !v10 )
      goto LABEL_24;
LABEL_3:
    v13 = v10 - 1;
    v14 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v15 = (__int64 *)(v12 + 16LL * v14);
    a5 = *v15;
    if ( v11 == *v15 )
    {
LABEL_4:
      v16 = v15[1];
      goto LABEL_5;
    }
    v26 = 1;
    while ( a5 != -4096 )
    {
      a6 = (unsigned int)(v26 + 1);
      v14 = v13 & (v26 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      a5 = *v15;
      if ( v11 == *v15 )
        goto LABEL_4;
      v26 = a6;
    }
LABEL_24:
    while ( 1 )
    {
      v16 = 0;
LABEL_5:
      if ( v7
        && v7 != v16
        && (*(_DWORD *)(v7 + 176) >= *(_DWORD *)(v16 + 176) || *(_DWORD *)(v7 + 180) <= *(_DWORD *)(v16 + 180)) )
      {
        do
        {
          v17 = *(unsigned int *)(v7 + 88);
          a5 = *(_QWORD *)(v7 + 160);
          a6 = *(_QWORD *)(v7 + 168);
          if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 92) )
          {
            v27 = *(_QWORD *)(v7 + 168);
            v28 = *(_QWORD *)(v7 + 160);
            sub_C8D5F0(v7 + 80, (const void *)(v7 + 96), v17 + 1, 0x10u, a5, a6);
            v17 = *(unsigned int *)(v7 + 88);
            a6 = v27;
            a5 = v28;
          }
          v18 = (__int64 *)(*(_QWORD *)(v7 + 80) + 16 * v17);
          *v18 = a6;
          v18[1] = a5;
          *(_QWORD *)(v7 + 168) = 0;
          ++*(_DWORD *)(v7 + 88);
          *(_QWORD *)(v7 + 160) = 0;
          v7 = *(_QWORD *)v7;
        }
        while ( v16 != v7
             && v7
             && (*(_DWORD *)(v7 + 176) >= *(_DWORD *)(v16 + 176) || *(_DWORD *)(v7 + 180) <= *(_DWORD *)(v16 + 180)) );
        v11 = *v8;
      }
      v19 = (_QWORD *)v16;
      do
      {
        while ( v19[21] )
        {
          v19 = (_QWORD *)*v19;
          if ( !v19 )
            goto LABEL_20;
        }
        v19[21] = v11;
        v19 = (_QWORD *)*v19;
      }
      while ( v19 );
LABEL_20:
      v20 = v8[1];
      v21 = (_QWORD *)v16;
      do
      {
        v21[20] = v20;
        v21 = (_QWORD *)*v21;
      }
      while ( v21 );
      v8 += 2;
      if ( (__int64 *)v9 == v8 )
        break;
      v10 = *(_DWORD *)(a3 + 24);
      v11 = *v8;
      v7 = v16;
      v12 = *(_QWORD *)(a3 + 8);
      if ( v10 )
        goto LABEL_3;
    }
    for ( ; v16; v16 = *(_QWORD *)v16 )
    {
      v23 = *(unsigned int *)(v16 + 88);
      v24 = *(_QWORD *)(v16 + 160);
      v25 = *(_QWORD *)(v16 + 168);
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 92) )
      {
        sub_C8D5F0(v16 + 80, (const void *)(v16 + 96), v23 + 1, 0x10u, a5, a6);
        v23 = *(unsigned int *)(v16 + 88);
      }
      v22 = (_QWORD *)(*(_QWORD *)(v16 + 80) + 16 * v23);
      *v22 = v25;
      v22[1] = v24;
      *(_QWORD *)(v16 + 168) = 0;
      ++*(_DWORD *)(v16 + 88);
      *(_QWORD *)(v16 + 160) = 0;
    }
  }
}
