// Function: sub_3891660
// Address: 0x3891660
//
unsigned __int64 __fastcall sub_3891660(_QWORD *a1, int *a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // r15
  unsigned int v6; // ebx
  __int64 v7; // rsi
  unsigned __int64 v8; // r13
  _QWORD *v9; // r8
  __int64 v10; // rax
  __int64 v11; // r12
  unsigned int v12; // edx
  __int64 v13; // rax
  char v14; // di
  __int64 v16; // rax
  _QWORD *v17; // [rsp+0h] [rbp-40h]

  v4 = sub_22077B0(0x40u);
  v5 = *((_QWORD *)a2 + 1);
  v6 = *a2;
  *((_QWORD *)a2 + 1) = 0;
  v7 = *((_QWORD *)a2 + 3);
  v8 = v4;
  v9 = a1 + 1;
  *((_QWORD *)a2 + 3) = 0;
  *(_DWORD *)(v4 + 32) = v6;
  *(_QWORD *)(v4 + 40) = v5;
  v10 = *((_QWORD *)a2 + 2);
  *((_QWORD *)a2 + 2) = 0;
  v11 = a1[2];
  *(_QWORD *)(v8 + 48) = v10;
  *(_QWORD *)(v8 + 56) = v7;
  if ( !v11 )
  {
    v11 = (__int64)(a1 + 1);
    if ( v9 == (_QWORD *)a1[3] )
    {
      v14 = 1;
      goto LABEL_10;
    }
LABEL_12:
    v17 = a1 + 1;
    v16 = sub_220EF80(v11);
    if ( *(_DWORD *)(v16 + 32) >= v6 )
    {
      v11 = v16;
      goto LABEL_14;
    }
    v9 = a1 + 1;
    if ( !v11 )
      goto LABEL_14;
    v14 = 1;
    if ( v17 != (_QWORD *)v11 )
      goto LABEL_19;
    goto LABEL_10;
  }
  while ( 1 )
  {
    v12 = *(_DWORD *)(v11 + 32);
    v13 = *(_QWORD *)(v11 + 24);
    if ( v12 > v6 )
      v13 = *(_QWORD *)(v11 + 16);
    if ( !v13 )
      break;
    v11 = v13;
  }
  if ( v6 < v12 )
  {
    if ( v11 != a1[3] )
      goto LABEL_12;
LABEL_9:
    v14 = 1;
    if ( v9 != (_QWORD *)v11 )
LABEL_19:
      v14 = v6 < *(_DWORD *)(v11 + 32);
LABEL_10:
    sub_220F040(v14, v8, (_QWORD *)v11, v9);
    ++a1[5];
    return v8;
  }
  if ( v12 < v6 )
    goto LABEL_9;
LABEL_14:
  if ( v5 )
    j_j___libc_free_0(v5);
  j_j___libc_free_0(v8);
  return v11;
}
