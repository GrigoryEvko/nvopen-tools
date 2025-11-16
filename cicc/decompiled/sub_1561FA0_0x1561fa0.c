// Function: sub_1561FA0
// Address: 0x1561fa0
//
__int64 __fastcall sub_1561FA0(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // r14
  __int64 v4; // r15
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v18; // [rsp+18h] [rbp-98h]
  _QWORD *v19; // [rsp+20h] [rbp-90h]
  __int64 v20; // [rsp+38h] [rbp-78h]
  __int64 v21[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v22[2]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD *v23; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v24[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( a2[7] )
    *(_QWORD *)(a1 + 56) = 0;
  if ( a2[8] )
    *(_QWORD *)(a1 + 64) = 0;
  if ( a2[9] )
    *(_QWORD *)(a1 + 72) = 0;
  if ( a2[10] )
    *(_QWORD *)(a1 + 80) = 0;
  if ( a2[11] )
    *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)a1 &= ~*a2 & 0x7FFFFFFFFFFFFFFLL;
  v3 = (_QWORD *)a2[4];
  v19 = a2 + 2;
  if ( v3 != a2 + 2 )
  {
    v4 = a1 + 16;
    v18 = a1 + 8;
    do
    {
      v5 = (_BYTE *)v3[4];
      v6 = (__int64)&v5[v3[5]];
      v21[0] = (__int64)v22;
      sub_155CB60(v21, v5, v6);
      v7 = (_BYTE *)v3[8];
      v8 = (__int64)&v7[v3[9]];
      v23 = v24;
      sub_155CB60((__int64 *)&v23, v7, v8);
      v9 = sub_1561D70(v18, (__int64)v21);
      v20 = v10;
      v11 = v9;
      if ( v9 == *(_QWORD *)(a1 + 32) && v10 == v4 )
      {
        sub_155CC10(*(_QWORD **)(a1 + 24));
        *(_QWORD *)(a1 + 32) = v4;
        *(_QWORD *)(a1 + 24) = 0;
        *(_QWORD *)(a1 + 40) = v4;
        *(_QWORD *)(a1 + 48) = 0;
      }
      else if ( v10 != v9 )
      {
        do
        {
          v12 = v11;
          v11 = sub_220EF30(v11);
          v13 = sub_220F330(v12, a1 + 16);
          v14 = *(_QWORD *)(v13 + 64);
          v15 = v13;
          if ( v14 != v13 + 80 )
            j_j___libc_free_0(v14, *(_QWORD *)(v13 + 80) + 1LL);
          v16 = *(_QWORD *)(v15 + 32);
          if ( v16 != v15 + 48 )
            j_j___libc_free_0(v16, *(_QWORD *)(v15 + 48) + 1LL);
          j_j___libc_free_0(v15, 96);
          --*(_QWORD *)(a1 + 48);
        }
        while ( v20 != v11 );
      }
      if ( v23 != v24 )
        j_j___libc_free_0(v23, v24[0] + 1LL);
      if ( (_QWORD *)v21[0] != v22 )
        j_j___libc_free_0(v21[0], v22[0] + 1LL);
      v3 = (_QWORD *)sub_220EF30(v3);
    }
    while ( v19 != v3 );
  }
  return a1;
}
