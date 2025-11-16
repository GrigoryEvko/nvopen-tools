// Function: sub_1C9E600
// Address: 0x1c9e600
//
__int64 __fastcall sub_1C9E600(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v2; // rbp
  _QWORD *v3; // rax
  _QWORD *v6; // r10
  unsigned __int64 v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 *v12[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = (_QWORD *)a1[2];
  v6 = a1 + 1;
  if ( !v3 )
  {
    v8 = a1 + 1;
LABEL_8:
    v12[1] = v2;
    v12[0] = a2;
    return sub_1C9E550(a1, v8, v12) + 40;
  }
  v7 = *a2;
  v8 = v6;
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
    v8 = v3;
    v3 = (_QWORD *)v3[2];
  }
  while ( v9 );
LABEL_6:
  if ( v6 == v8 || v8[4] > v7 )
    goto LABEL_8;
  return (__int64)(v8 + 5);
}
