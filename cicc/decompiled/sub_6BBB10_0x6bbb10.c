// Function: sub_6BBB10
// Address: 0x6bbb10
//
__int64 __fastcall sub_6BBB10(_QWORD *a1)
{
  __int64 *v2; // rax
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 *v8; // rdx
  _QWORD *v9; // r15
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 *v13; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v14; // [rsp+8h] [rbp-D8h] BYREF
  _BYTE v15[208]; // [rsp+10h] [rbp-D0h] BYREF

  v2 = *(__int64 **)(*a1 + 24LL);
  v3 = qword_4F06BC0;
  v13 = v2;
  v4 = v2[3];
  if ( (v2[5] & 1) != 0 )
  {
    do
    {
      v8 = v2;
      v2 = (__int64 *)*v2;
    }
    while ( v2 );
    v9 = *(_QWORD **)(v8[1] + 24);
    v10 = (_QWORD *)*v9;
    *v9 = 0;
    v11 = v10;
    do
    {
      v12 = v11;
      v11 = (_QWORD *)*v11;
    }
    while ( v11 != a1 );
    *v12 = 0;
    sub_6E1990(v10);
    *v9 = a1;
    v2 = v13;
  }
  qword_4F06BC0 = v2[4];
  sub_6E2250(v15, &v14, 4, 1, v4, 0);
  sub_6BA760(1, (__int64)&v13);
  sub_6E2C70(v14, 1, v4, 0);
  v5 = v13;
  if ( v13 )
  {
    v6 = qword_4F06BC0;
    v13[3] = v4;
    v5[4] = v6;
  }
  else
  {
    *(_BYTE *)(v4 + 179) &= ~2u;
  }
  qword_4F06BC0 = v3;
  return *a1;
}
