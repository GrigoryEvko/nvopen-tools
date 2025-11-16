// Function: sub_30A8900
// Address: 0x30a8900
//
_QWORD *__fastcall sub_30A8900(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r12
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdx
  _QWORD *v6; // rax
  _BOOL4 v7; // r8d
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r9
  char v12; // r8
  __int64 v13; // rbx
  __int64 v15; // rax
  _BOOL4 v16; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  if ( v3 )
  {
    v4 = *a2;
    while ( 1 )
    {
      v5 = v3[4];
      v6 = (_QWORD *)v3[3];
      if ( v4 < v5 )
        v6 = (_QWORD *)v3[2];
      if ( !v6 )
        break;
      v3 = v6;
    }
    if ( v4 >= v5 )
    {
      if ( v4 > v5 )
        goto LABEL_9;
      return v3;
    }
    if ( v3 == (_QWORD *)a1[3] )
    {
LABEL_9:
      v7 = 1;
      if ( v2 != v3 )
        v7 = v4 < v3[4];
      goto LABEL_11;
    }
LABEL_15:
    v15 = sub_220EF80((__int64)v3);
    if ( v4 <= *(_QWORD *)(v15 + 32) )
      return (_QWORD *)v15;
    goto LABEL_9;
  }
  v3 = a1 + 1;
  if ( v2 != (_QWORD *)a1[3] )
  {
    v4 = *a2;
    goto LABEL_15;
  }
  v7 = 1;
LABEL_11:
  v16 = v7;
  v8 = (_QWORD *)sub_22077B0(0x40u);
  v12 = v16;
  v13 = (__int64)v8;
  v8[4] = *a2;
  v8[5] = v8 + 7;
  v8[6] = 0x100000000LL;
  if ( *((_DWORD *)a2 + 4) )
  {
    sub_30A6A60((__int64)(v8 + 5), (char **)a2 + 1, v9, v10, v16, v11);
    v12 = v16;
  }
  sub_220F040(v12, v13, v3, v2);
  ++a1[5];
  return (_QWORD *)v13;
}
