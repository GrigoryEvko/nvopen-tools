// Function: sub_12AA280
// Address: 0x12aa280
//
_QWORD *__fastcall sub_12AA280(__int64 *a1, int a2, int a3, int a4, int a5, int a6, unsigned __int8 a7)
{
  __int64 v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rdi
  unsigned __int64 *v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v21; // [rsp+18h] [rbp-58h] BYREF
  char v22[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v23; // [rsp+30h] [rbp-40h]

  v23 = 257;
  v9 = sub_1648A60(64, 3);
  v10 = (_QWORD *)v9;
  if ( v9 )
    sub_15F99E0(v9, a2, a3, a4, a5, a6, a7, 0);
  v11 = a1[1];
  if ( v11 )
  {
    v12 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v11 + 40, v10);
    v13 = v10[3];
    v14 = *v12;
    v10[4] = v12;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v10 + 3;
    *v12 = *v12 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v10, v22);
  v15 = *a1;
  if ( *a1 )
  {
    v21 = *a1;
    sub_1623A60(&v21, v15, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v16 = v21;
    v10[6] = v21;
    if ( v16 )
      sub_1623210(&v21, v16, v10 + 6);
  }
  return v10;
}
