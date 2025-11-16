// Function: sub_25D9B40
// Address: 0x25d9b40
//
_QWORD *__fastcall sub_25D9B40(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rcx
  _QWORD *v5; // rax
  char v6; // si
  unsigned __int64 v7; // rdx
  bool v8; // zf
  __int64 v10; // rax
  unsigned __int64 v11; // rdx

  v3 = *(_QWORD **)(a1 + 16);
  if ( !v3 )
  {
    v3 = (_QWORD *)(a1 + 8);
LABEL_13:
    if ( v3 != *(_QWORD **)(a1 + 24) )
    {
      v10 = sub_220EF80((__int64)v3);
      v11 = *(_QWORD *)(v10 + 32);
      v3 = (_QWORD *)v10;
      v8 = v11 == *a2;
      if ( v11 >= *a2 )
        goto LABEL_15;
    }
    return 0;
  }
  v4 = *a2;
  while ( 1 )
  {
    v7 = v3[4];
    if ( v7 > v4 || v7 == v4 && v3[5] > a2[1] )
      break;
    v5 = (_QWORD *)v3[3];
    v6 = 0;
    if ( !v5 )
      goto LABEL_9;
LABEL_6:
    v3 = v5;
  }
  v5 = (_QWORD *)v3[2];
  v6 = 1;
  if ( v5 )
    goto LABEL_6;
LABEL_9:
  if ( v6 )
    goto LABEL_13;
  v8 = v7 == v4;
  if ( v7 < v4 )
    return 0;
LABEL_15:
  if ( v8 && v3[5] < a2[1] )
    return 0;
  return v3;
}
