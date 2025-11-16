// Function: sub_C929D0
// Address: 0xc929d0
//
__int64 __fastcall sub_C929D0(__int64 *a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 *v3; // rbx
  unsigned int v4; // r14d
  int v5; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // r15
  bool v13; // zf
  __int64 v14; // r11
  unsigned int v15; // r14d
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned int v18; // r9d
  __int64 v19; // r8
  int v20; // r10d
  unsigned int v21; // eax
  __int64 v22; // rcx
  int v23; // edx
  int v24; // ecx
  __int64 v25; // rax
  unsigned int v27; // [rsp+14h] [rbp-3Ch]

  v2 = a2;
  v3 = a1;
  v4 = *((_DWORD *)a1 + 2);
  v5 = *((_DWORD *)a1 + 3);
  if ( 4 * v5 > 3 * v4 )
  {
    v27 = 2 * v4;
  }
  else
  {
    if ( v4 - (*((_DWORD *)a1 + 4) + v5) > v4 >> 3 )
      return v2;
    v27 = *((_DWORD *)a1 + 2);
  }
  v11 = _libc_calloc(v27 + 1, 12);
  if ( !v11 )
  {
    if ( v27 != -1 )
      sub_C64F00("Allocation failed", 1u);
    v25 = sub_C65340(1, 12, v7, v8, v9, v10);
    v4 = *((_DWORD *)a1 + 2);
    v11 = v25;
  }
  v12 = v4;
  v13 = v4 == 0;
  v14 = *a1;
  v15 = a2;
  *(_QWORD *)(v11 + 8LL * v27) = 2;
  v16 = v27;
  if ( !v13 )
  {
    v17 = 0;
    v18 = v27 - 1;
    do
    {
      v19 = *(_QWORD *)(v14 + 8 * v17);
      if ( v19 && v19 != -8 )
      {
        v20 = *(_DWORD *)(v14 + 8 * v12 + 8 + 4 * v17);
        v21 = v20 & v18;
        v22 = v20 & v18;
        v16 = v11 + 8 * v22;
        if ( *(_QWORD *)v16 )
        {
          v23 = 1;
          do
          {
            v24 = v23++;
            v21 = v18 & (v24 + v21);
            v22 = v21;
            v16 = v11 + 8LL * v21;
          }
          while ( *(_QWORD *)v16 );
        }
        *(_QWORD *)v16 = v19;
        *(_DWORD *)(v11 + 4 * v22 + 8LL * v27 + 8) = v20;
        if ( v2 == (_DWORD)v17 )
          v15 = v21;
      }
      ++v17;
    }
    while ( v12 != v17 );
    v3 = a1;
  }
  v2 = v15;
  _libc_free(v14, v16);
  *v3 = v11;
  *((_DWORD *)v3 + 4) = 0;
  *((_DWORD *)v3 + 2) = v27;
  return v2;
}
