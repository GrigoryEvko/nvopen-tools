// Function: sub_1E99A80
// Address: 0x1e99a80
//
_QWORD *__fastcall sub_1E99A80(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r12
  __int64 v4; // r13
  int v5; // ecx
  int v6; // edx
  _QWORD *v7; // rax
  _BOOL4 v8; // r8d
  __int64 v9; // r13
  __int64 v11; // rax
  _BOOL4 v12; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  if ( !v3 )
  {
    v3 = a1 + 1;
    if ( v2 == (_QWORD *)a1[3] )
    {
      v8 = 1;
LABEL_11:
      v12 = v8;
      v9 = sub_22077B0(40);
      *(_QWORD *)(v9 + 32) = *a2;
      sub_220F040(v12, v9, v3, v2);
      ++a1[5];
      return (_QWORD *)v9;
    }
    v4 = *a2;
LABEL_13:
    v11 = sub_220EF80(v3);
    if ( *(_DWORD *)(v4 + 48) <= *(_DWORD *)(*(_QWORD *)(v11 + 32) + 48LL) )
      return (_QWORD *)v11;
LABEL_9:
    v8 = 1;
    if ( v2 != v3 )
      v8 = *(_DWORD *)(v4 + 48) < *(_DWORD *)(v3[4] + 48LL);
    goto LABEL_11;
  }
  v4 = *a2;
  v5 = *(_DWORD *)(*a2 + 48);
  while ( 1 )
  {
    v6 = *(_DWORD *)(v3[4] + 48LL);
    v7 = (_QWORD *)v3[3];
    if ( v5 < v6 )
      v7 = (_QWORD *)v3[2];
    if ( !v7 )
      break;
    v3 = v7;
  }
  if ( v5 < v6 )
  {
    if ( (_QWORD *)a1[3] == v3 )
      goto LABEL_9;
    goto LABEL_13;
  }
  if ( v5 > v6 )
    goto LABEL_9;
  return v3;
}
