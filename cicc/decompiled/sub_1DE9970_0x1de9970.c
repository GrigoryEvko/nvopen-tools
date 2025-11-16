// Function: sub_1DE9970
// Address: 0x1de9970
//
__int64 *__fastcall sub_1DE9970(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r10
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rax
  int v13; // r11d
  __int64 v14; // r10
  int v15; // eax
  __int64 *v16; // rax
  __int64 v17; // rdx
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // r10
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 *v26; // rax
  int v27; // r11d
  __int64 v28; // r10
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rcx
  int v32; // esi
  __int64 v33; // rdi

  if ( a4 > a5 && a5 <= a7 )
  {
    if ( !a5 )
      return a1;
    v7 = a3 - a2;
    v8 = a2 - (_QWORD)a1;
    v9 = (a3 - a2) >> 4;
    v10 = (a2 - (__int64)a1) >> 4;
    if ( a3 - a2 <= 0 )
    {
      if ( v8 <= 0 )
        return a1;
      v14 = 0;
      v7 = 0;
    }
    else
    {
      v11 = a6;
      v12 = a2;
      do
      {
        v13 = *(_DWORD *)(v12 + 8);
        v11 += 16;
        v12 += 16;
        *(_DWORD *)(v11 - 8) = v13;
        *(_QWORD *)(v11 - 16) = *(_QWORD *)(v12 - 16);
        --v9;
      }
      while ( v9 );
      if ( v7 <= 0 )
        v7 = 16;
      v14 = v7 >> 4;
      if ( v8 <= 0 )
      {
LABEL_11:
        if ( v7 > 0 )
        {
          v16 = a1;
          v17 = v14;
          do
          {
            v18 = *(_DWORD *)(a6 + 8);
            v16 += 2;
            a6 += 16;
            *((_DWORD *)v16 - 2) = v18;
            *(v16 - 2) = *(_QWORD *)(a6 - 16);
            --v17;
          }
          while ( v17 );
          v19 = 2 * v14;
          if ( v14 <= 0 )
            v19 = 2;
          return &a1[v19];
        }
        return a1;
      }
    }
    do
    {
      v15 = *(_DWORD *)(a2 - 8);
      a2 -= 16;
      a3 -= 16;
      *(_DWORD *)(a3 + 8) = v15;
      *(_QWORD *)a3 = *(_QWORD *)a2;
      --v10;
    }
    while ( v10 );
    goto LABEL_11;
  }
  if ( a4 > a7 )
    return sub_1DE36B0(a1, (__int64 *)a2, (__int64 *)a3);
  if ( !a4 )
    return (__int64 *)a3;
  v21 = a2 - (_QWORD)a1;
  v22 = a3 - a2;
  v23 = (a2 - (__int64)a1) >> 4;
  v24 = (a3 - a2) >> 4;
  if ( a2 - (__int64)a1 <= 0 )
  {
    if ( v22 <= 0 )
      return (__int64 *)a3;
    v28 = 0;
    v21 = 0;
    goto LABEL_25;
  }
  v25 = a6;
  v26 = a1;
  do
  {
    v27 = *((_DWORD *)v26 + 2);
    v25 += 16;
    v26 += 2;
    *(_DWORD *)(v25 - 8) = v27;
    *(_QWORD *)(v25 - 16) = *(v26 - 2);
    --v23;
  }
  while ( v23 );
  if ( v21 <= 0 )
    v21 = 16;
  a6 += v21;
  v28 = v21 >> 4;
  if ( v22 > 0 )
  {
    do
    {
LABEL_25:
      v29 = *(_DWORD *)(a2 + 8);
      a1 += 2;
      a2 += 16;
      *((_DWORD *)a1 - 2) = v29;
      *(a1 - 2) = *(_QWORD *)(a2 - 16);
      --v24;
    }
    while ( v24 );
  }
  if ( v21 <= 0 )
    return (__int64 *)a3;
  v30 = a3;
  v31 = v28;
  do
  {
    v32 = *(_DWORD *)(a6 - 8);
    a6 -= 16;
    v30 -= 16;
    *(_DWORD *)(v30 + 8) = v32;
    *(_QWORD *)v30 = *(_QWORD *)a6;
    --v31;
  }
  while ( v31 );
  v33 = -16 * v28;
  if ( v28 <= 0 )
    v33 = -16;
  return (__int64 *)(a3 + v33);
}
