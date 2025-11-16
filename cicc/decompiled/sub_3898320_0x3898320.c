// Function: sub_3898320
// Address: 0x3898320
//
unsigned __int64 *__fastcall sub_3898320(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v2; // rbp
  _QWORD *v3; // rax
  _QWORD *v6; // r10
  unsigned __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 *v12[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = (_QWORD *)a1[2];
  v6 = a1 + 1;
  if ( !v3 )
  {
    v8 = (__int64)(a1 + 1);
LABEL_8:
    v12[1] = v2;
    v12[0] = a2;
    return sub_3898250(a1, v8, v12) + 5;
  }
  v7 = *a2;
  v8 = (__int64)v6;
  do
  {
    while ( 1 )
    {
      v9 = v3[2];
      v10 = v3[3];
      if ( v3[4] >= v7 )
        break;
      v3 = (_QWORD *)v3[3];
      if ( !v10 )
        goto LABEL_6;
    }
    v8 = (__int64)v3;
    v3 = (_QWORD *)v3[2];
  }
  while ( v9 );
LABEL_6:
  if ( v6 == (_QWORD *)v8 || *(_QWORD *)(v8 + 32) > v7 )
    goto LABEL_8;
  return (unsigned __int64 *)(v8 + 40);
}
