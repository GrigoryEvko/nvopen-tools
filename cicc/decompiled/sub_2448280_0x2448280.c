// Function: sub_2448280
// Address: 0x2448280
//
void __fastcall sub_2448280(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v5; // r12
  bool v6; // zf
  _QWORD *v7; // rax
  unsigned __int64 v8; // rdx
  _QWORD *i; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  int v12; // esi
  int v13; // r10d
  __int64 *v14; // r9
  unsigned int v15; // edx
  __int64 *v16; // rdi
  __int64 v17; // r8
  _QWORD *v18; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  int v21; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_QWORD **)(a1 + 16);
    v8 = (unsigned __int64)*(unsigned int *)(a1 + 24) << 6;
  }
  else
  {
    v7 = (_QWORD *)(a1 + 16);
    v8 = 256;
  }
  for ( i = (_QWORD *)((char *)v7 + v8); i != v7; v7 += 8 )
  {
    if ( v7 )
      *v7 = -4096;
  }
  if ( a2 != a3 )
  {
    do
    {
      v10 = *v5;
      if ( *v5 != -4096 && v10 != -8192 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v11 = a1 + 16;
          v12 = 3;
        }
        else
        {
          v21 = *(_DWORD *)(a1 + 24);
          v11 = *(_QWORD *)(a1 + 16);
          if ( !v21 )
          {
            MEMORY[0] = *v5;
            BUG();
          }
          v12 = v21 - 1;
        }
        v13 = 1;
        v14 = 0;
        v15 = v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v16 = (__int64 *)(v11 + ((unsigned __int64)v15 << 6));
        v17 = *v16;
        if ( v10 != *v16 )
        {
          while ( v17 != -4096 )
          {
            if ( v17 == -8192 && !v14 )
              v14 = v16;
            v15 = v12 & (v13 + v15);
            v16 = (__int64 *)(v11 + ((unsigned __int64)v15 << 6));
            v17 = *v16;
            if ( v10 == *v16 )
              goto LABEL_13;
            ++v13;
          }
          if ( v14 )
            v16 = v14;
        }
LABEL_13:
        *v16 = v10;
        sub_24481E0((__m128i *)(v16 + 1), (__m128i *)(v5 + 1));
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v18 = (_QWORD *)v5[3];
        while ( v18 )
        {
          v19 = (unsigned __int64)v18;
          v18 = (_QWORD *)*v18;
          j_j___libc_free_0(v19);
        }
        memset((void *)v5[1], 0, 8 * v5[2]);
        v20 = v5[1];
        v5[4] = 0;
        v5[3] = 0;
        if ( (__int64 *)v20 != v5 + 7 )
          j_j___libc_free_0(v20);
      }
      v5 += 8;
    }
    while ( a3 != v5 );
  }
}
