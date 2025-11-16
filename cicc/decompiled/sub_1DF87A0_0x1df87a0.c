// Function: sub_1DF87A0
// Address: 0x1df87a0
//
__int64 __fastcall sub_1DF87A0(_QWORD *a1, unsigned int *a2)
{
  _QWORD *v2; // r8
  _QWORD *v3; // r12
  unsigned int v4; // ebx
  unsigned int v5; // edx
  _QWORD *v6; // rax
  _BOOL4 v7; // r9d
  __int64 v8; // r15
  __int64 v10; // rax
  _BOOL4 v11; // [rsp+4h] [rbp-3Ch]
  _QWORD *v12; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  if ( v3 )
  {
    v4 = *a2;
    while ( 1 )
    {
      v5 = *((_DWORD *)v3 + 8);
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
      return (__int64)v3;
    }
    if ( v3 == (_QWORD *)a1[3] )
    {
LABEL_9:
      v7 = 1;
      if ( v2 != v3 )
        v7 = v4 < *((_DWORD *)v3 + 8);
      goto LABEL_11;
    }
LABEL_14:
    v10 = sub_220EF80(v3);
    v2 = a1 + 1;
    if ( v4 <= *(_DWORD *)(v10 + 32) )
      return v10;
    goto LABEL_9;
  }
  v3 = a1 + 1;
  if ( v2 != (_QWORD *)a1[3] )
  {
    v4 = *a2;
    goto LABEL_14;
  }
  v7 = 1;
LABEL_11:
  v11 = v7;
  v12 = v2;
  v8 = sub_22077B0(40);
  *(_DWORD *)(v8 + 32) = *a2;
  sub_220F040(v11, v8, v3, v12);
  ++a1[5];
  return v8;
}
