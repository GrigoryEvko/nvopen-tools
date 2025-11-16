// Function: sub_35D0870
// Address: 0x35d0870
//
__int64 __fastcall sub_35D0870(__int64 a1, unsigned int *a2, unsigned int *a3)
{
  unsigned int *v5; // rbx
  bool v6; // zf
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 i; // rdx
  unsigned int *v10; // r13
  __int64 v11; // r9
  __int64 v12; // r8
  int v13; // r10d
  unsigned int v14; // edx
  unsigned int *v15; // rsi
  unsigned int *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned int *v21; // r13
  unsigned __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rdi
  int v25; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD *)(a1 + 16);
    v8 = 56LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v8 = 224;
  }
  for ( i = result + v8; i != result; result += 56 )
  {
    if ( result )
      *(_DWORD *)result = 0x7FFFFFFF;
  }
  if ( a2 != a3 )
  {
    do
    {
      while ( 1 )
      {
        result = *v5;
        v10 = v5 + 14;
        if ( (unsigned int)(result + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v5 += 14;
        if ( a3 == v10 )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v11 = a1 + 16;
        v12 = 3;
      }
      else
      {
        v25 = *(_DWORD *)(a1 + 24);
        v11 = *(_QWORD *)(a1 + 16);
        if ( !v25 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v12 = (unsigned int)(v25 - 1);
      }
      v13 = 1;
      v14 = v12 & (37 * result);
      v15 = 0;
      v16 = (unsigned int *)(v11 + 56LL * v14);
      v17 = *v16;
      if ( (_DWORD)result != (_DWORD)v17 )
      {
        while ( (_DWORD)v17 != 0x7FFFFFFF )
        {
          if ( (_DWORD)v17 == 0x80000000 && !v15 )
            v15 = v16;
          v14 = v12 & (v13 + v14);
          v16 = (unsigned int *)(v11 + 56LL * v14);
          v17 = *v16;
          if ( (_DWORD)result == (_DWORD)v17 )
            goto LABEL_14;
          ++v13;
        }
        if ( v15 )
          v16 = v15;
      }
LABEL_14:
      *((_QWORD *)v16 + 3) = 0;
      *((_QWORD *)v16 + 2) = 0;
      v16[8] = 0;
      *v16 = result;
      *((_QWORD *)v16 + 1) = 1;
      v18 = *((_QWORD *)v5 + 2);
      ++*((_QWORD *)v5 + 1);
      v19 = *((_QWORD *)v16 + 2);
      *((_QWORD *)v16 + 2) = v18;
      LODWORD(v18) = v5[6];
      *((_QWORD *)v5 + 2) = v19;
      LODWORD(v19) = v16[6];
      v16[6] = v18;
      LODWORD(v18) = v5[7];
      v5[6] = v19;
      LODWORD(v19) = v16[7];
      v16[7] = v18;
      v20 = v5[8];
      v5[7] = v19;
      LODWORD(v19) = v16[8];
      v16[8] = v20;
      v5[8] = v19;
      *((_QWORD *)v16 + 5) = v16 + 14;
      *((_QWORD *)v16 + 6) = 0;
      if ( v5[12] )
        sub_35CFBA0((__int64)(v16 + 10), (char **)v5 + 5, v20, v17, v12, v11);
      v21 = v5 + 14;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v22 = *((_QWORD *)v5 + 5);
      if ( (unsigned int *)v22 != v5 + 14 )
        _libc_free(v22);
      v23 = v5[8];
      v24 = *((_QWORD *)v5 + 2);
      v5 += 14;
      result = sub_C7D6A0(v24, 8 * v23, 8);
    }
    while ( a3 != v21 );
  }
  return result;
}
