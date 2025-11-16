// Function: sub_2DAF240
// Address: 0x2daf240
//
unsigned int *__fastcall sub_2DAF240(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r15
  unsigned int v3; // r14d
  __int64 v4; // r13
  __int64 *v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  unsigned int *result; // rax
  unsigned int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rbx
  __int64 *v17; // r13
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx

  v1 = 0;
  v2 = *(unsigned int *)(*a1 + 64);
  if ( (_DWORD)v2 )
  {
    do
    {
      v3 = v1;
      v4 = v1++;
      v3 |= 0x80000000;
      v5 = (__int64 *)(a1[2] + 32 * v4);
      v5[2] = sub_2DAEAE0(a1, v3);
      v5[3] = v6;
      *v5 = sub_2DAE650(a1, v3, v6, v7, v8, v9);
      v5[1] = v10;
    }
    while ( v2 != v1 );
  }
  for ( result = (unsigned int *)a1[5]; result != (unsigned int *)a1[9]; result = (unsigned int *)a1[5] )
  {
    v12 = *result;
    if ( result == (unsigned int *)(a1[7] - 4) )
    {
      j_j___libc_free_0(a1[6]);
      v20 = (__int64 *)(a1[8] + 8);
      a1[8] = (__int64)v20;
      v21 = *v20;
      v22 = *v20 + 512;
      a1[6] = v21;
      a1[7] = v22;
      a1[5] = v21;
    }
    else
    {
      a1[5] = (__int64)(result + 1);
    }
    v13 = v12 >> 6;
    v14 = 1LL << v12;
    v15 = v12;
    v16 = 16LL * (v12 & 0x7FFFFFFF);
    *(_QWORD *)(a1[13] + 8 * v13) &= ~v14;
    v17 = (__int64 *)(a1[2] + 32 * v15);
    v18 = *(_QWORD *)(*(_QWORD *)(*a1 + 56) + v16 + 8);
    if ( v18 )
    {
      if ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
      {
        v18 = *(_QWORD *)(v18 + 32);
        if ( v18 )
        {
          if ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
            BUG();
        }
      }
    }
    sub_2DAF190(a1, *(_QWORD *)(v18 + 16), *v17, v17[1]);
    v19 = *(_QWORD *)(*(_QWORD *)(*a1 + 56) + v16 + 8);
    if ( v19 )
    {
      while ( (*(_BYTE *)(v19 + 3) & 0x10) != 0 || (*(_BYTE *)(v19 + 4) & 8) != 0 )
      {
        v19 = *(_QWORD *)(v19 + 32);
        if ( !v19 )
          goto LABEL_18;
      }
LABEL_13:
      sub_2DAE870((__int64)a1, v19, v17[2], v17[3]);
      while ( 1 )
      {
        v19 = *(_QWORD *)(v19 + 32);
        if ( !v19 )
          break;
        while ( (*(_BYTE *)(v19 + 3) & 0x10) == 0 )
        {
          if ( (*(_BYTE *)(v19 + 4) & 8) == 0 )
            goto LABEL_13;
          v19 = *(_QWORD *)(v19 + 32);
          if ( !v19 )
            goto LABEL_18;
        }
      }
    }
LABEL_18:
    ;
  }
  return result;
}
