// Function: sub_31BE530
// Address: 0x31be530
//
void __fastcall sub_31BE530(__int64 a1)
{
  unsigned int v2; // r14d
  int v3; // r13d
  _QWORD *v4; // rbx
  _QWORD *v5; // r15
  __int64 v6; // rdi
  int v7; // eax
  unsigned int v8; // r13d
  __int64 v9; // rbx
  _QWORD *v10; // rdi
  unsigned int v11; // ebx
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  _QWORD *v16; // rax

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_DWORD *)(a1 + 16);
  if ( !v2 )
  {
    if ( !v3 )
      goto LABEL_31;
    v8 = v3 - 1;
    if ( !v8 )
    {
      v10 = *(_QWORD **)(a1 + 8);
      LODWORD(v9) = 64;
LABEL_18:
      sub_C7D6A0((__int64)v10, 16LL * v2, 8);
      v11 = 4 * (int)v9 / 3u;
      v12 = ((((((((((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
                 | (v11 + 1)
                 | ((unsigned __int64)(v11 + 1) >> 1)) >> 4)
               | (((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
               | (v11 + 1)
               | ((unsigned __int64)(v11 + 1) >> 1)) >> 8)
             | (((((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2) | (v11 + 1)
                                                                         | ((unsigned __int64)(v11 + 1) >> 1)) >> 4)
             | (((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
             | (v11 + 1)
             | ((unsigned __int64)(v11 + 1) >> 1)) >> 16)
           | (((((((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2) | (v11 + 1)
                                                                         | ((unsigned __int64)(v11 + 1) >> 1)) >> 4)
             | (((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
             | (v11 + 1)
             | ((unsigned __int64)(v11 + 1) >> 1)) >> 8)
           | (((((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2) | (v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 4)
           | (((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
           | (v11 + 1)
           | ((unsigned __int64)(v11 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v12;
      v13 = (_QWORD *)sub_C7D670(16 * v12, 8);
      v14 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = v13;
      for ( i = &v13[2 * v14]; i != v13; v13 += 2 )
      {
        if ( v13 )
          *v13 = -4096;
      }
      return;
    }
    v7 = 0;
    goto LABEL_15;
  }
  v4 = *(_QWORD **)(a1 + 8);
  v5 = &v4[2 * v2];
  do
  {
    if ( *v4 != -4096 && *v4 != -8192 )
    {
      v6 = v4[1];
      if ( v6 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    }
    v4 += 2;
  }
  while ( v5 != v4 );
  v7 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    if ( v7 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * v2, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
LABEL_31:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v9 = 64;
  v8 = v3 - 1;
  if ( v8 )
  {
LABEL_15:
    _BitScanReverse(&v8, v8);
    v9 = (unsigned int)(1 << (33 - (v8 ^ 0x1F)));
    if ( (int)v9 < 64 )
      v9 = 64;
  }
  v10 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v9 != v7 )
    goto LABEL_18;
  *(_QWORD *)(a1 + 16) = 0;
  v16 = &v10[2 * v9];
  do
  {
    if ( v10 )
      *v10 = -4096;
    v10 += 2;
  }
  while ( v16 != v10 );
}
