// Function: sub_8896D0
// Address: 0x8896d0
//
void __fastcall sub_8896D0(unsigned __int64 a1)
{
  __int64 v1; // rbx
  int v2; // r9d
  __int64 v3; // r8
  unsigned int v4; // esi
  unsigned int v5; // edx
  __int64 v6; // r10
  __int64 v7; // r12
  __int64 v8; // r11
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // r10
  __int64 v14; // r14
  int v15; // eax
  unsigned int v16; // r13d
  unsigned int v17; // r12d
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r9
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rsi
  unsigned __int64 v24; // rdi
  unsigned __int64 i; // rdx
  unsigned int v26; // edx
  __int64 v27; // rax
  __int64 v28; // r14
  int v29; // eax
  _QWORD *v30; // rax
  __int64 v31; // rsi
  unsigned __int64 v32; // rdi
  unsigned __int64 j; // rdx
  unsigned int v34; // edx
  __int64 v35; // rax

  if ( dword_4D03FE8[0] )
  {
    *(_BYTE *)(a1 + 73) |= 0x40u;
    return;
  }
  v1 = qword_4F600F8;
  v2 = *(_DWORD *)(qword_4F600F8 + 8);
  v3 = *(_QWORD *)qword_4F600F8;
  v4 = v2 & (a1 >> 3);
  v5 = v4;
  v6 = 16LL * v4;
  v7 = *(_QWORD *)qword_4F600F8 + v6;
  v8 = *(_QWORD *)v7;
  v9 = *(_QWORD *)v7;
  if ( a1 != *(_QWORD *)v7 )
  {
    while ( v9 )
    {
      v5 = v2 & (v5 + 1);
      v10 = v3 + 16LL * v5;
      v9 = *(_QWORD *)v10;
      if ( a1 == *(_QWORD *)v10 )
        goto LABEL_5;
    }
LABEL_9:
    if ( v8 )
    {
      do
      {
        v4 = v2 & (v4 + 1);
        v11 = v3 + 16LL * v4;
      }
      while ( *(_QWORD *)v11 );
      v12 = *(_DWORD *)(v7 + 8);
      *(_QWORD *)v11 = v8;
      *(_DWORD *)(v11 + 8) = v12;
      *(_QWORD *)v7 = 0;
      v13 = *(_QWORD *)v1 + v6;
      *(_QWORD *)v13 = a1;
      if ( a1 )
        *(_DWORD *)(v13 + 8) = 1;
      v14 = *(unsigned int *)(v1 + 8);
      v15 = *(_DWORD *)(v1 + 12) + 1;
      *(_DWORD *)(v1 + 12) = v15;
      if ( 2 * v15 <= (unsigned int)v14 )
        return;
      v16 = v14 + 1;
      v17 = 2 * v14 + 1;
      v18 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v14 + 2));
      v21 = (__int64)v18;
      if ( 2 * (_DWORD)v14 != -2 )
      {
        v19 = (__int64)&v18[2 * v17 + 2];
        do
        {
          if ( v18 )
            *v18 = 0;
          v18 += 2;
        }
        while ( (_QWORD *)v19 != v18 );
      }
      v22 = *(_QWORD *)v1;
      if ( (_DWORD)v14 != -1 )
      {
        v23 = *(_QWORD *)v1;
        v20 = (__int64 *)(v22 + 16 * v14 + 16);
        do
        {
          v24 = *(_QWORD *)v23;
          if ( *(_QWORD *)v23 )
          {
            for ( i = v24 >> 3; ; LODWORD(i) = v26 + 1 )
            {
              v26 = v17 & i;
              v27 = v21 + 16LL * v26;
              if ( !*(_QWORD *)v27 )
                break;
            }
            *(_QWORD *)v27 = v24;
            v19 = *(unsigned int *)(v23 + 8);
            *(_DWORD *)(v27 + 8) = v19;
          }
          v23 += 16;
        }
        while ( v20 != (__int64 *)v23 );
      }
    }
    else
    {
      *(_QWORD *)v7 = a1;
      if ( a1 )
        *(_DWORD *)(v7 + 8) = 1;
      v28 = *(unsigned int *)(v1 + 8);
      v29 = *(_DWORD *)(v1 + 12) + 1;
      *(_DWORD *)(v1 + 12) = v29;
      if ( 2 * v29 <= (unsigned int)v28 )
        return;
      v16 = v28 + 1;
      v17 = 2 * v28 + 1;
      v30 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v28 + 2));
      v21 = (__int64)v30;
      if ( 2 * (_DWORD)v28 != -2 )
      {
        v19 = (__int64)&v30[2 * v17 + 2];
        do
        {
          if ( v30 )
            *v30 = 0;
          v30 += 2;
        }
        while ( (_QWORD *)v19 != v30 );
      }
      v22 = *(_QWORD *)v1;
      if ( (_DWORD)v28 != -1 )
      {
        v31 = *(_QWORD *)v1;
        v20 = (__int64 *)(v22 + 16 * v28 + 16);
        do
        {
          v32 = *(_QWORD *)v31;
          if ( *(_QWORD *)v31 )
          {
            for ( j = v32 >> 3; ; LODWORD(j) = v34 + 1 )
            {
              v34 = v17 & j;
              v35 = v21 + 16LL * v34;
              if ( !*(_QWORD *)v35 )
                break;
            }
            *(_QWORD *)v35 = v32;
            v19 = *(unsigned int *)(v31 + 8);
            *(_DWORD *)(v35 + 8) = v19;
          }
          v31 += 16;
        }
        while ( (__int64 *)v31 != v20 );
      }
    }
    *(_QWORD *)v1 = v21;
    *(_DWORD *)(v1 + 8) = v17;
    sub_823A00(v22, 16LL * v16, v19, v21, v22, v20);
    return;
  }
  v10 = *(_QWORD *)qword_4F600F8 + v6;
LABEL_5:
  if ( !*(_DWORD *)(v10 + 8) )
    goto LABEL_9;
}
