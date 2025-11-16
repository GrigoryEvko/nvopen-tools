// Function: sub_2EB38F0
// Address: 0x2eb38f0
//
bool __fastcall sub_2EB38F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 *v10; // r8
  __int64 *v11; // r13
  __int64 v12; // r10
  __int64 *i; // rdi
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 *v18; // rsi
  __int64 v19; // rcx
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 *v23; // rax
  __int64 **v24; // r12
  __int64 v25; // r13
  __int64 **v26; // r14
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rsi
  __int64 *v30; // rdi
  __int64 v31; // rax
  _QWORD *v32; // rax
  _QWORD *v33; // rcx
  __int64 j; // rdx

  if ( *(_QWORD *)(a1 + 128) != *(_QWORD *)(a2 + 128) )
    return 1;
  v9 = *(unsigned int *)(a1 + 8);
  if ( v9 != *(_DWORD *)(a2 + 8) )
    return 1;
  v10 = *(__int64 **)a1;
  v11 = *(__int64 **)a2;
  v12 = *(_QWORD *)a1 + 8 * v9;
  if ( v12 != *(_QWORD *)a1 )
  {
    while ( *v10 == *v11 )
    {
      ++v10;
      ++v11;
      if ( (__int64 *)v12 == v10 )
        goto LABEL_45;
    }
    if ( (__int64 *)v12 != v10 )
    {
      a6 = *v10;
      for ( i = v10; ; a6 = *i )
      {
        v14 = ((char *)i - (char *)v10) >> 5;
        v15 = i - v10;
        if ( v14 > 0 )
        {
          v16 = *i;
          a4 = *v10;
          v17 = v10;
          v18 = &v10[4 * v14];
          while ( v16 != a4 )
          {
            if ( v16 == v17[1] )
            {
              ++v17;
              break;
            }
            if ( v16 == v17[2] )
            {
              v17 += 2;
              break;
            }
            if ( v16 == v17[3] )
            {
              v17 += 3;
              break;
            }
            v17 += 4;
            if ( v17 == v18 )
            {
              v15 = i - v17;
              goto LABEL_24;
            }
            a4 = *v17;
          }
LABEL_20:
          if ( v17 != i )
            goto LABEL_21;
          goto LABEL_28;
        }
        v17 = v10;
LABEL_24:
        if ( v15 != 2 )
        {
          if ( v15 != 3 )
          {
            if ( v15 != 1 )
              goto LABEL_28;
            goto LABEL_27;
          }
          a4 = *i;
          if ( *v17 == *i )
            goto LABEL_20;
          ++v17;
        }
        a4 = *i;
        if ( *v17 == *i )
          goto LABEL_20;
        ++v17;
LABEL_27:
        a4 = *i;
        if ( *v17 == *i )
          goto LABEL_20;
LABEL_28:
        v19 = *v11;
        v20 = v11;
        v21 = 0;
        while ( 1 )
        {
          ++v20;
          v21 += a6 == v19;
          if ( v20 == (__int64 *)((char *)v11 + v12 - (_QWORD)v10) )
            break;
          v19 = *v20;
        }
        if ( !v21 )
          return 1;
        if ( (__int64 *)v12 == i )
          return 1;
        v22 = a6;
        v23 = i;
        a4 = 0;
        while ( 1 )
        {
          ++v23;
          a4 += a6 == v22;
          if ( (__int64 *)v12 == v23 )
            break;
          v22 = *v23;
        }
        if ( v21 != a4 )
          return 1;
LABEL_21:
        if ( (__int64 *)v12 == ++i )
          break;
      }
    }
  }
LABEL_45:
  v24 = *(__int64 ***)(a1 + 48);
  v25 = 0;
  v26 = &v24[*(unsigned int *)(a1 + 56)];
  if ( v26 != v24 )
  {
    while ( 1 )
    {
      v30 = *v24;
      if ( *v24 )
      {
        v31 = *v30;
        if ( *v30 )
        {
          v27 = (unsigned int)(*(_DWORD *)(v31 + 24) + 1);
          v28 = *(_DWORD *)(v31 + 24) + 1;
        }
        else
        {
          v27 = 0;
          v28 = 0;
        }
        v29 = 0;
        if ( v28 < *(_DWORD *)(a2 + 56) )
          v29 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v27);
        if ( (unsigned __int8)sub_2E6C990((__int64)v30, v29, v27, a4, (__int64)v10, a6) )
          return 1;
        ++v25;
      }
      if ( v26 == ++v24 )
        goto LABEL_56;
    }
  }
  v25 = 0;
LABEL_56:
  v32 = *(_QWORD **)(a2 + 48);
  v33 = &v32[*(unsigned int *)(a2 + 56)];
  for ( j = 0; v33 != v32; ++v32 )
    j -= (*v32 == 0) - 1LL;
  return v25 != j;
}
