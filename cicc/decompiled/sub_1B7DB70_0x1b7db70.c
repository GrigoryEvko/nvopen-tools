// Function: sub_1B7DB70
// Address: 0x1b7db70
//
_QWORD *__fastcall sub_1B7DB70(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // rcx
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r13
  _QWORD *v5; // rax
  char v6; // dl
  unsigned __int64 v7; // r12
  bool v8; // zf
  _QWORD *result; // rax
  int v10; // eax
  __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // [rsp+8h] [rbp-38h]

  v2 = a1;
  v3 = *(_QWORD **)(a1 + 16);
  if ( !v3 )
  {
    v3 = (_QWORD *)(a1 + 8);
    goto LABEL_17;
  }
  v4 = *a2;
  while ( 1 )
  {
    v7 = v3[4];
    if ( v4 < v7 )
    {
      v5 = (_QWORD *)v3[2];
      v6 = 1;
      goto LABEL_8;
    }
    if ( v4 == v7 )
    {
      v13 = v2;
      v10 = sub_16A9900((__int64)(a2 + 1), v3 + 5);
      v2 = v13;
      if ( v10 < 0 )
        break;
    }
    v5 = (_QWORD *)v3[3];
    v6 = 0;
    if ( !v5 )
      goto LABEL_9;
LABEL_5:
    v3 = v5;
  }
  v5 = (_QWORD *)v3[2];
  v6 = 1;
LABEL_8:
  if ( v5 )
    goto LABEL_5;
LABEL_9:
  if ( !v6 )
  {
    v8 = v7 == v4;
    if ( v7 < v4 )
      return 0;
LABEL_11:
    if ( v8 && (int)sub_16A9900((__int64)(v3 + 5), a2 + 1) < 0 )
      return 0;
    else
      return v3;
  }
LABEL_17:
  result = 0;
  if ( v3 != *(_QWORD **)(v2 + 24) )
  {
    v11 = sub_220EF80(v3);
    v12 = *(_QWORD *)(v11 + 32);
    v3 = (_QWORD *)v11;
    v8 = v12 == *a2;
    if ( v12 >= *a2 )
      goto LABEL_11;
    return 0;
  }
  return result;
}
