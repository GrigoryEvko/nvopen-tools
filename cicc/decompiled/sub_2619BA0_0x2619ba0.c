// Function: sub_2619BA0
// Address: 0x2619ba0
//
_QWORD *__fastcall sub_2619BA0(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r14
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdx
  _QWORD *v5; // rax
  char v6; // r15
  _QWORD *v7; // r13
  unsigned __int64 v8; // rax
  __int64 v10; // rax
  _QWORD v11[2]; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v12; // [rsp+10h] [rbp-40h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  v11[0] = v11;
  v11[1] = 1;
  v12 = a2;
  if ( v3 )
  {
    while ( 1 )
    {
      v4 = v3[6];
      v5 = (_QWORD *)v3[3];
      if ( v4 > a2 )
        v5 = (_QWORD *)v3[2];
      if ( !v5 )
        break;
      v3 = v5;
    }
    if ( a2 >= v4 )
    {
      if ( a2 <= v4 )
        return v3;
LABEL_9:
      v6 = 1;
      if ( v2 != v3 )
        v6 = v12 < v3[6];
      goto LABEL_11;
    }
    if ( (_QWORD *)a1[3] == v3 )
      goto LABEL_9;
LABEL_14:
    v10 = sub_220EF80((__int64)v3);
    if ( *(_QWORD *)(v10 + 48) >= v12 )
      return (_QWORD *)v10;
    goto LABEL_9;
  }
  v3 = a1 + 1;
  if ( v2 != (_QWORD *)a1[3] )
    goto LABEL_14;
  v6 = 1;
LABEL_11:
  v7 = (_QWORD *)sub_22077B0(0x38u);
  v7[4] = v7 + 4;
  v8 = v12;
  v7[5] = 1;
  v7[6] = v8;
  sub_220F040(v6, (__int64)v7, v3, v2);
  ++a1[5];
  return v7;
}
