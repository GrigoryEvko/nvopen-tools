// Function: sub_12A94B0
// Address: 0x12a94b0
//
_QWORD *__fastcall sub_12A94B0(__int64 *a1, int a2, int a3, int a4, int a5, unsigned __int8 a6)
{
  __int64 v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rdi
  unsigned __int64 *v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  char v21[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v22 = 257;
  v9 = sub_1648A60(64, 2);
  v10 = (_QWORD *)v9;
  if ( v9 )
    sub_15F9C10(v9, a2, a3, a4, a5, a6, 0);
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
  sub_164B780(v10, v21);
  v15 = *a1;
  if ( *a1 )
  {
    v20 = *a1;
    sub_1623A60(&v20, v15, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v16 = v20;
    v10[6] = v20;
    if ( v16 )
      sub_1623210(&v20, v16, v10 + 6);
  }
  return v10;
}
