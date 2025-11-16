// Function: sub_B20CA0
// Address: 0xb20ca0
//
__int64 __fastcall sub_B20CA0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdx
  int v4; // edi
  __int64 v5; // rcx
  __int64 v6; // r12
  unsigned __int64 v7; // r15
  __int64 v8; // r13
  int v9; // eax
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 i; // rdx
  __int64 v13; // rax
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 536);
  if ( !a2 )
  {
    if ( !(_DWORD)v3 )
    {
      v7 = 1;
      v8 = a1 + 528;
      v6 = 0;
      v10 = 56;
      if ( *(_DWORD *)(a1 + 540) )
        goto LABEL_8;
      goto LABEL_18;
    }
    v6 = 0;
LABEL_15:
    v13 = *(_QWORD *)(a1 + 528);
    return v6 + v13;
  }
  v4 = *(_DWORD *)(a2 + 44);
  v5 = (unsigned int)(v4 + 1);
  v6 = 56 * v5;
  if ( (unsigned int)v5 < (unsigned int)v3 )
    goto LABEL_15;
  v7 = (unsigned int)(v4 + 2);
  v8 = a1 + 528;
  v9 = *(_DWORD *)(*(_QWORD *)(a2 + 72) + 88LL);
  if ( v9 )
    v7 = (unsigned int)(v9 + 1);
  if ( v7 == v3 )
    goto LABEL_15;
  v10 = 56 * v7;
  if ( v7 >= v3 )
  {
    if ( *(unsigned int *)(a1 + 540) >= v7 )
    {
LABEL_8:
      v11 = *(_QWORD *)v8 + 56LL * *(unsigned int *)(v8 + 8);
      for ( i = v10 + *(_QWORD *)v8; i != v11; v11 += 56 )
      {
        if ( v11 )
        {
          *(_QWORD *)(v11 + 48) = 0;
          *(_OWORD *)(v11 + 16) = 0;
          *(_OWORD *)(v11 + 32) = 0;
          *(_QWORD *)(v11 + 24) = v11 + 40;
          *(_DWORD *)(v11 + 36) = 4;
          *(_OWORD *)v11 = 0;
        }
      }
      *(_DWORD *)(v8 + 8) = v7;
      v13 = *(_QWORD *)(a1 + 528);
      return v6 + v13;
    }
LABEL_18:
    v18 = v10;
    sub_B20B60(v8, v7);
    v10 = v18;
    goto LABEL_8;
  }
  v13 = *(_QWORD *)(a1 + 528);
  v15 = v13 + 56 * v3;
  v16 = v13 + 56 * v7;
  if ( v15 != v16 )
  {
    do
    {
      v15 -= 56;
      v17 = *(_QWORD *)(v15 + 24);
      if ( v17 != v15 + 40 )
        _libc_free(v17, a2);
    }
    while ( v16 != v15 );
    v13 = *(_QWORD *)(a1 + 528);
  }
  *(_DWORD *)(a1 + 536) = v7;
  return v6 + v13;
}
