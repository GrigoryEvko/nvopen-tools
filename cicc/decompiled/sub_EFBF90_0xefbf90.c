// Function: sub_EFBF90
// Address: 0xefbf90
//
__int64 __fastcall sub_EFBF90(__int64 a1, __int64 a2, char a3)
{
  unsigned __int64 v4; // rax
  _QWORD *v5; // rbx
  __int64 i; // r14
  unsigned __int64 v7; // r15
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  _BOOL8 v15; // rdi
  __int64 result; // rax
  __int64 v17; // r14
  __int64 j; // r12
  _QWORD *v19; // [rsp+18h] [rbp-38h]
  _QWORD *v20; // [rsp+18h] [rbp-38h]

  if ( a3 )
  {
    result = a2;
    if ( (*(_BYTE *)(a2 + 52) & 4) != 0 )
      return result;
  }
  else
  {
    ++*(_DWORD *)(a1 + 124);
    v4 = *(_QWORD *)(a2 + 64);
    if ( *(_QWORD *)(a1 + 112) < v4 )
      *(_QWORD *)(a1 + 112) = v4;
  }
  v5 = (_QWORD *)(a1 + 8);
  for ( i = *(_QWORD *)(a2 + 96); a2 + 80 != i; i = sub_220EF30(i) )
  {
    v7 = *(_QWORD *)(i + 40);
    *(_QWORD *)(a1 + 96) += v7;
    if ( v7 > *(_QWORD *)(a1 + 104) )
      *(_QWORD *)(a1 + 104) = v7;
    v8 = *(_QWORD **)(a1 + 16);
    ++*(_DWORD *)(a1 + 120);
    v9 = (_QWORD *)(a1 + 8);
    if ( !v8 )
      goto LABEL_14;
    do
    {
      while ( 1 )
      {
        v10 = v8[2];
        v11 = v8[3];
        if ( v7 >= v8[4] )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v11 )
          goto LABEL_12;
      }
      v9 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( v10 );
LABEL_12:
    if ( v5 == v9 || v7 > v9[4] )
    {
LABEL_14:
      v19 = v9;
      v12 = sub_22077B0(48);
      *(_QWORD *)(v12 + 32) = v7;
      v9 = (_QWORD *)v12;
      *(_DWORD *)(v12 + 40) = 0;
      v13 = sub_EFBD70((_QWORD *)a1, v19, (unsigned __int64 *)(v12 + 32));
      if ( v14 )
      {
        v15 = v5 == v14 || v13 || v7 > v14[4];
        sub_220F040(v15, v9, v14, a1 + 8);
        ++*(_QWORD *)(a1 + 40);
      }
      else
      {
        v20 = v13;
        j_j___libc_free_0(v9, 48);
        v9 = v20;
      }
    }
    ++*((_DWORD *)v9 + 10);
  }
  result = a2;
  v17 = *(_QWORD *)(a2 + 144);
  if ( a2 + 128 != v17 )
  {
    do
    {
      for ( j = *(_QWORD *)(v17 + 64); v17 + 48 != j; j = sub_220EF30(j) )
        sub_EFBF90(a1, j + 48, 1);
      result = sub_220EF30(v17);
      v17 = result;
    }
    while ( a2 + 128 != result );
  }
  return result;
}
