// Function: sub_1AD5AF0
// Address: 0x1ad5af0
//
__int64 __fastcall sub_1AD5AF0(__int64 a1, _QWORD *a2)
{
  int v2; // eax
  int v3; // ebx
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rax
  _BYTE *v8; // rsi
  _BYTE *v9; // rbx
  __int64 *v10; // r13
  __int64 *v11; // rax
  __int64 v12; // r12
  _QWORD *v14; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v15; // [rsp+10h] [rbp-40h] BYREF
  _BYTE *v16; // [rsp+18h] [rbp-38h]
  _BYTE *v17; // [rsp+20h] [rbp-30h]

  v2 = sub_1CCB210();
  v14 = a2;
  v3 = v2;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  sub_1273E00((__int64)&v15, 0, &v14);
  v4 = (_QWORD *)sub_15E0530(a1);
  v5 = sub_1643350(v4);
  v6 = sub_159C470(v5, v3, 0);
  v7 = sub_1624210(v6);
  v8 = v16;
  v14 = v7;
  if ( v16 == v17 )
  {
    sub_1273E00((__int64)&v15, v16, &v14);
    v9 = v16;
  }
  else
  {
    if ( v16 )
    {
      *(_QWORD *)v16 = v7;
      v8 = v16;
    }
    v9 = v8 + 8;
    v16 = v8 + 8;
  }
  v10 = v15;
  v11 = (__int64 *)sub_15E0530(a1);
  v12 = sub_1627350(v11, v10, (__int64 *)((v9 - (_BYTE *)v10) >> 3), 0, 1);
  if ( v15 )
    j_j___libc_free_0(v15, v17 - (_BYTE *)v15);
  return v12;
}
