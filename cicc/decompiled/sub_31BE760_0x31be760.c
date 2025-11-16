// Function: sub_31BE760
// Address: 0x31be760
//
void __fastcall sub_31BE760(__int64 a1)
{
  unsigned int v2; // r12d
  int v3; // ebx
  _QWORD *v4; // r14
  _QWORD *v5; // r13
  unsigned __int64 v6; // r8
  __int64 *v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v9; // rdx
  int v10; // eax
  unsigned int v11; // ebx
  __int64 v12; // r13
  _QWORD *v13; // rdi
  unsigned int v14; // r13d
  unsigned __int64 v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  _QWORD *v19; // rax
  unsigned __int64 v20; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_DWORD *)(a1 + 16);
  if ( !v2 )
  {
    if ( !v3 )
      goto LABEL_36;
    v11 = v3 - 1;
    if ( !v11 )
    {
      v13 = *(_QWORD **)(a1 + 8);
      LODWORD(v12) = 64;
LABEL_23:
      sub_C7D6A0((__int64)v13, 16LL * v2, 8);
      v14 = 4 * (int)v12 / 3u;
      v15 = ((((((((((v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 2)
                 | (v14 + 1)
                 | ((unsigned __int64)(v14 + 1) >> 1)) >> 4)
               | (((v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 2)
               | (v14 + 1)
               | ((unsigned __int64)(v14 + 1) >> 1)) >> 8)
             | (((((v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 2) | (v14 + 1)
                                                                         | ((unsigned __int64)(v14 + 1) >> 1)) >> 4)
             | (((v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 2)
             | (v14 + 1)
             | ((unsigned __int64)(v14 + 1) >> 1)) >> 16)
           | (((((((v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 2) | (v14 + 1)
                                                                         | ((unsigned __int64)(v14 + 1) >> 1)) >> 4)
             | (((v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 2)
             | (v14 + 1)
             | ((unsigned __int64)(v14 + 1) >> 1)) >> 8)
           | (((((v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 2) | (v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 4)
           | (((v14 + 1) | ((unsigned __int64)(v14 + 1) >> 1)) >> 2)
           | (v14 + 1)
           | ((unsigned __int64)(v14 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v15;
      v16 = (_QWORD *)sub_C7D670(16 * v15, 8);
      v17 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = v16;
      for ( i = &v16[2 * v17]; i != v16; v16 += 2 )
      {
        if ( v16 )
          *v16 = -4096;
      }
      return;
    }
    v10 = 0;
    goto LABEL_20;
  }
  v4 = *(_QWORD **)(a1 + 8);
  v5 = &v4[2 * v2];
  do
  {
    if ( *v4 != -4096 && *v4 != -8192 )
    {
      v6 = v4[1];
      if ( v6 )
      {
        v7 = *(__int64 **)v6;
        v8 = *(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8);
        if ( *(_QWORD *)v6 != v8 )
        {
          do
          {
            v9 = *v7++;
            *(_QWORD *)(v9 + 32) = 0;
          }
          while ( v7 != (__int64 *)v8 );
          v8 = *(_QWORD *)v6;
        }
        if ( v8 != v6 + 16 )
        {
          v20 = v6;
          _libc_free(v8);
          v6 = v20;
        }
        j_j___libc_free_0(v6);
      }
    }
    v4 += 2;
  }
  while ( v5 != v4 );
  v10 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    if ( v10 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * v2, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
LABEL_36:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v12 = 64;
  v11 = v3 - 1;
  if ( v11 )
  {
LABEL_20:
    _BitScanReverse(&v11, v11);
    v12 = (unsigned int)(1 << (33 - (v11 ^ 0x1F)));
    if ( (int)v12 < 64 )
      v12 = 64;
  }
  v13 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v12 != v10 )
    goto LABEL_23;
  *(_QWORD *)(a1 + 16) = 0;
  v19 = &v13[2 * v12];
  do
  {
    if ( v13 )
      *v13 = -4096;
    v13 += 2;
  }
  while ( v19 != v13 );
}
