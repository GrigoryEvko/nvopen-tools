// Function: sub_2E6F1C0
// Address: 0x2e6f1c0
//
__int64 __fastcall sub_2E6F1C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  int v8; // edi
  __int64 v9; // rcx
  __int64 v10; // r12
  unsigned __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 i; // rdx
  __int64 v17; // rax
  __int64 v19; // r13
  __int64 v20; // r14
  unsigned __int64 v21; // rdi
  __int64 v22; // [rsp+8h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 536);
  if ( !a2 )
  {
    if ( !(_DWORD)v7 )
    {
      v11 = 1;
      v12 = a1 + 528;
      v10 = 0;
      v14 = 56;
      if ( *(_DWORD *)(a1 + 540) )
        goto LABEL_8;
      goto LABEL_18;
    }
    v10 = 0;
LABEL_15:
    v17 = *(_QWORD *)(a1 + 528);
    return v10 + v17;
  }
  v8 = *(_DWORD *)(a2 + 24);
  v9 = (unsigned int)(v8 + 1);
  v10 = 56 * v9;
  if ( (unsigned int)v9 < (unsigned int)v7 )
    goto LABEL_15;
  v11 = (unsigned int)(v8 + 2);
  v12 = a1 + 528;
  v13 = (__int64)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 104LL) - *(_QWORD *)(*(_QWORD *)(a2 + 32) + 96LL)) >> 3;
  a4 = (unsigned int)(v13 + 1);
  if ( (_DWORD)v13 )
    v11 = (unsigned int)(v13 + 1);
  if ( v11 == v7 )
    goto LABEL_15;
  v14 = 56 * v11;
  if ( v11 >= v7 )
  {
    if ( v11 <= *(unsigned int *)(a1 + 540) )
    {
LABEL_8:
      v15 = *(_QWORD *)v12 + 56LL * *(unsigned int *)(v12 + 8);
      for ( i = v14 + *(_QWORD *)v12; i != v15; v15 += 56 )
      {
        if ( v15 )
        {
          *(_QWORD *)(v15 + 48) = 0;
          *(_OWORD *)(v15 + 16) = 0;
          *(_OWORD *)(v15 + 32) = 0;
          *(_QWORD *)(v15 + 24) = v15 + 40;
          *(_DWORD *)(v15 + 36) = 4;
          *(_OWORD *)v15 = 0;
        }
      }
      *(_DWORD *)(v12 + 8) = v11;
      v17 = *(_QWORD *)(a1 + 528);
      return v10 + v17;
    }
LABEL_18:
    v22 = v14;
    sub_2E6EFC0(v12, v11, v7, a4, v14, a6);
    v14 = v22;
    goto LABEL_8;
  }
  v17 = *(_QWORD *)(a1 + 528);
  v19 = v17 + 56 * v7;
  v20 = v17 + 56 * v11;
  if ( v19 != v20 )
  {
    do
    {
      v19 -= 56;
      v21 = *(_QWORD *)(v19 + 24);
      if ( v21 != v19 + 40 )
        _libc_free(v21);
    }
    while ( v20 != v19 );
    v17 = *(_QWORD *)(a1 + 528);
  }
  *(_DWORD *)(a1 + 536) = v11;
  return v10 + v17;
}
