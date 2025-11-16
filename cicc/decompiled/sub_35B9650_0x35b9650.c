// Function: sub_35B9650
// Address: 0x35b9650
//
void __fastcall sub_35B9650(unsigned int *a1, unsigned int *a2)
{
  unsigned int v3; // eax
  unsigned __int64 v5; // r13
  void *v6; // rax
  void *v7; // rcx
  unsigned __int64 v8; // r13
  void *v9; // rax
  void *v10; // rcx
  __int64 v11; // r13
  unsigned int *v12; // rdi
  __int64 v13; // rcx
  unsigned int v14; // r10d
  unsigned int v15; // r11d
  unsigned int v16; // esi
  unsigned int v17; // r8d
  unsigned int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned int v22; // edx
  unsigned int *i; // rax

  v3 = *a2;
  *(_QWORD *)a1 = 0;
  v5 = v3 - 1;
  v6 = (void *)sub_2207820(v5);
  v7 = v6;
  if ( v6 && v5 )
    v7 = memset(v6, 0, v5);
  *((_QWORD *)a1 + 1) = v7;
  v8 = a2[1] - 1;
  v9 = (void *)sub_2207820(v8);
  v10 = v9;
  if ( v9 && v8 )
    v10 = memset(v9, 0, v8);
  *((_QWORD *)a1 + 2) = v10;
  v11 = a2[1] - 1;
  v12 = (unsigned int *)sub_2207820(4 * v11);
  if ( v12 && v11 )
    v12 = (unsigned int *)memset(v12, 0, 4 * v11);
  v13 = a2[1];
  if ( *a2 > 1 )
  {
    v14 = *a1;
    v15 = 0;
    v16 = 1;
    do
    {
      if ( (unsigned int)v13 > 1 )
      {
        v17 = 0;
        v18 = 1;
        do
        {
          if ( *(float *)(*((_QWORD *)a2 + 1) + 4 * (v18 + (unsigned __int64)(v16 * (unsigned int)v13))) == INFINITY )
          {
            v19 = *((_QWORD *)a1 + 1);
            v20 = v18 - 1;
            ++v17;
            ++v12[v20];
            *(_BYTE *)(v19 + v15) = 1;
            *(_BYTE *)(*((_QWORD *)a1 + 2) + v20) = 1;
            v13 = a2[1];
          }
          ++v18;
        }
        while ( (unsigned int)v13 > v18 );
        if ( v14 < v17 )
          v14 = v17;
      }
      *a1 = v14;
      ++v16;
      ++v15;
    }
    while ( *a2 > v16 );
  }
  v21 = (__int64)&v12[v13 - 1];
  v22 = *v12;
  if ( (unsigned int *)v21 != v12 )
  {
    for ( i = v12 + 1; (unsigned int *)v21 != i; ++i )
    {
      if ( v22 < *i )
        v22 = *i;
    }
  }
  if ( a1[1] >= v22 )
    v22 = a1[1];
  a1[1] = v22;
  j_j___libc_free_0_0((unsigned __int64)v12);
}
