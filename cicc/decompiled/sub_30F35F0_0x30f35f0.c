// Function: sub_30F35F0
// Address: 0x30f35f0
//
_QWORD *__fastcall sub_30F35F0(__int64 a1)
{
  _QWORD **v1; // rax
  __int64 v2; // r8
  _QWORD *v3; // r9
  _QWORD **v4; // r8
  _QWORD **v5; // rsi
  _QWORD *v6; // rdi
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  unsigned int v10; // edx

  v1 = *(_QWORD ***)a1;
  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD **)(*(_QWORD *)a1 + v2 * 8 - 8);
  if ( !*v3 )
    return v3;
  v4 = &v1[v2];
  if ( v1 == v4 )
    return v3;
  v5 = v1 + 1;
  if ( v4 == v1 + 1 )
    return v3;
  v6 = (_QWORD *)**v1;
  while ( 1 )
  {
    v7 = v6;
    v6 = (_QWORD *)**v5;
    if ( !v6 )
      break;
    v8 = (_QWORD *)**v5;
    v9 = 1;
    do
    {
      v8 = (_QWORD *)*v8;
      ++v9;
    }
    while ( v8 );
    if ( v7 )
      goto LABEL_9;
    if ( !v9 )
      goto LABEL_17;
LABEL_12:
    if ( v4 == ++v5 )
      return v3;
  }
  if ( !v7 )
    goto LABEL_12;
  v9 = 1;
LABEL_9:
  v10 = 1;
  do
  {
    v7 = (_QWORD *)*v7;
    ++v10;
  }
  while ( v7 );
  if ( v10 <= v9 )
    goto LABEL_12;
LABEL_17:
  if ( v4 != v5 )
    return 0;
  return v3;
}
