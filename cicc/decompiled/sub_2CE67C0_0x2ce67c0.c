// Function: sub_2CE67C0
// Address: 0x2ce67c0
//
__int64 __fastcall sub_2CE67C0(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *v2; // r14
  _QWORD *v5; // rax
  unsigned __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  char v14; // di
  unsigned __int64 v16; // rdi
  __int64 v17; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v5 = (_QWORD *)a1[2];
  if ( !v5 )
  {
    v7 = (__int64)(a1 + 1);
LABEL_8:
    v17 = v7;
    v10 = sub_22077B0(0x30u);
    v11 = *a2;
    *(_DWORD *)(v10 + 40) = 0;
    v7 = v10;
    *(_QWORD *)(v10 + 32) = v11;
    v12 = sub_2CBBD90(a1, v17, (unsigned __int64 *)(v10 + 32));
    if ( v13 )
    {
      v14 = v12 || v2 == v13 || v11 < v13[4];
      sub_220F040(v14, v7, v13, v2);
      ++a1[5];
    }
    else
    {
      v16 = v7;
      v7 = (__int64)v12;
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
      v8 = v5[2];
      v9 = v5[3];
      if ( v5[4] >= v6 )
        break;
      v5 = (_QWORD *)v5[3];
      if ( !v9 )
        goto LABEL_6;
    }
    v7 = (__int64)v5;
    v5 = (_QWORD *)v5[2];
  }
  while ( v8 );
LABEL_6:
  if ( v2 == (_QWORD *)v7 || *(_QWORD *)(v7 + 32) > v6 )
    goto LABEL_8;
  return v7 + 40;
}
