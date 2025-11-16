// Function: sub_38B2690
// Address: 0x38b2690
//
__int64 __fastcall sub_38B2690(_QWORD *a1, unsigned int *a2)
{
  _QWORD *v2; // r14
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // r12
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rdx
  char v14; // di
  unsigned __int64 v16; // rdi
  __int64 v17; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v5 = a1[2];
  if ( !v5 )
  {
    v7 = (__int64)(a1 + 1);
LABEL_8:
    v17 = v7;
    v10 = sub_22077B0(0x38u);
    v11 = *a2;
    *(_QWORD *)(v10 + 40) = 0;
    v7 = v10;
    *(_DWORD *)(v10 + 32) = v11;
    *(_QWORD *)(v10 + 48) = 0;
    v12 = sub_38B2590(a1, v17, (unsigned int *)(v10 + 32));
    if ( v13 )
    {
      v14 = v12 || v2 == (_QWORD *)v13 || v11 < *(_DWORD *)(v13 + 32);
      sub_220F040(v14, v7, (_QWORD *)v13, v2);
      ++a1[5];
    }
    else
    {
      v16 = v7;
      v7 = v12;
      j_j___libc_free_0(v16);
    }
    return v7 + 40;
  }
  v6 = *a2;
  v7 = (__int64)(a1 + 1);
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v5 + 16);
      v9 = *(_QWORD *)(v5 + 24);
      if ( *(_DWORD *)(v5 + 32) >= v6 )
        break;
      v5 = *(_QWORD *)(v5 + 24);
      if ( !v9 )
        goto LABEL_6;
    }
    v7 = v5;
    v5 = *(_QWORD *)(v5 + 16);
  }
  while ( v8 );
LABEL_6:
  if ( v2 == (_QWORD *)v7 || v6 < *(_DWORD *)(v7 + 32) )
    goto LABEL_8;
  return v7 + 40;
}
