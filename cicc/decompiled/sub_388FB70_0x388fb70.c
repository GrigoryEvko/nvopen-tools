// Function: sub_388FB70
// Address: 0x388fb70
//
unsigned __int64 __fastcall sub_388FB70(_QWORD *a1, unsigned int *a2)
{
  _QWORD *v3; // r15
  __int64 v4; // rax
  unsigned int v5; // ebx
  unsigned __int64 v6; // r13
  __int64 v7; // r12
  unsigned int v8; // edx
  __int64 v9; // rax
  char v10; // di
  __int64 v12; // rax

  v3 = a1 + 1;
  v4 = sub_22077B0(0x30u);
  v5 = *a2;
  v6 = v4;
  *(_DWORD *)(v4 + 32) = *a2;
  v7 = a1[2];
  *(_QWORD *)(v4 + 40) = *((_QWORD *)a2 + 1);
  if ( !v7 )
  {
    v7 = (__int64)(a1 + 1);
    if ( v3 == (_QWORD *)a1[3] )
    {
      v10 = 1;
      goto LABEL_10;
    }
LABEL_12:
    v12 = sub_220EF80(v7);
    if ( v5 <= *(_DWORD *)(v12 + 32) )
    {
      v7 = v12;
      goto LABEL_14;
    }
    if ( !v7 )
      goto LABEL_14;
    v10 = 1;
    if ( (_QWORD *)v7 != v3 )
      goto LABEL_17;
    goto LABEL_10;
  }
  while ( 1 )
  {
    v8 = *(_DWORD *)(v7 + 32);
    v9 = *(_QWORD *)(v7 + 24);
    if ( v5 < v8 )
      v9 = *(_QWORD *)(v7 + 16);
    if ( !v9 )
      break;
    v7 = v9;
  }
  if ( v5 < v8 )
  {
    if ( v7 != a1[3] )
      goto LABEL_12;
LABEL_9:
    v10 = 1;
    if ( (_QWORD *)v7 != v3 )
LABEL_17:
      v10 = v5 < *(_DWORD *)(v7 + 32);
LABEL_10:
    sub_220F040(v10, v6, (_QWORD *)v7, a1 + 1);
    ++a1[5];
    return v6;
  }
  if ( v5 > v8 )
    goto LABEL_9;
LABEL_14:
  j_j___libc_free_0(v6);
  return v7;
}
