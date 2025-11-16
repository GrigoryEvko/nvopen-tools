// Function: sub_1CB7560
// Address: 0x1cb7560
//
_QWORD *__fastcall sub_1CB7560(_QWORD *a1, unsigned __int64 a2, int a3)
{
  _QWORD *v3; // r15
  _QWORD *v4; // r14
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  _QWORD *v11; // rcx
  _BOOL4 v12; // r8d
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v16; // rax
  _BOOL4 v17; // [rsp+4h] [rbp-5Ch]
  _QWORD *v18; // [rsp+8h] [rbp-58h]
  _QWORD *v19; // [rsp+10h] [rbp-50h]
  unsigned __int64 v20; // [rsp+18h] [rbp-48h] BYREF
  unsigned __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  int v22; // [rsp+28h] [rbp-38h]

  v3 = a1 + 2;
  v4 = a1 + 2;
  v6 = (_QWORD *)a1[3];
  v20 = a2;
  if ( v6 )
  {
    do
    {
      while ( 1 )
      {
        v7 = v6[2];
        v8 = v6[3];
        if ( v6[4] >= a2 )
          break;
        v6 = (_QWORD *)v6[3];
        if ( !v8 )
          goto LABEL_6;
      }
      v4 = v6;
      v6 = (_QWORD *)v6[2];
    }
    while ( v7 );
LABEL_6:
    if ( v3 != v4 )
    {
      if ( v4[4] <= a2 )
      {
        v16 = sub_1819210((__int64)(a1 + 13), &v20);
        if ( !v10 )
          goto LABEL_12;
        v11 = a1 + 14;
        v12 = 1;
        if ( v16 )
          goto LABEL_11;
        goto LABEL_16;
      }
      v4 = a1 + 2;
    }
  }
  v9 = sub_1819210((__int64)(a1 + 13), &v20);
  if ( !v10 )
    goto LABEL_13;
  v11 = a1 + 14;
  v12 = 1;
  if ( !v9 )
  {
LABEL_16:
    if ( v10 != v11 )
      v12 = v20 < v10[4];
  }
LABEL_11:
  v17 = v12;
  v18 = v11;
  v19 = v10;
  v13 = sub_22077B0(40);
  *(_QWORD *)(v13 + 32) = v20;
  sub_220F040(v17, v13, v19, v18);
  ++a1[18];
  if ( v3 != v4 )
  {
LABEL_12:
    v14 = sub_220F330(v4, v3);
    j_j___libc_free_0(v14, 48);
    --a1[6];
  }
LABEL_13:
  v22 = a3;
  v21 = v20;
  return sub_1CB74B0((__int64)(a1 + 1), &v21);
}
