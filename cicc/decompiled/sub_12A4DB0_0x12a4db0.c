// Function: sub_12A4DB0
// Address: 0x12a4db0
//
_QWORD *__fastcall sub_12A4DB0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rdi
  unsigned __int64 *v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v18; // rdx
  __int64 v19; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  v21 = 257;
  v9 = sub_1648A60(56, 3);
  v10 = (_QWORD *)v9;
  if ( v9 )
    sub_15F83E0(v9, a3, a4, a2, 0);
  v11 = a1[7];
  if ( v11 )
  {
    v12 = (unsigned __int64 *)a1[8];
    sub_157E9D0(v11 + 40, v10);
    v13 = v10[3];
    v14 = *v12;
    v10[4] = v12;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v10 + 3;
    *v12 = *v12 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v10, v20);
  v15 = a1[6];
  if ( v15 )
  {
    v19 = a1[6];
    sub_1623A60(&v19, v15, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v16 = v19;
    v10[6] = v19;
    if ( v16 )
      sub_1623210(&v19, v16, v10 + 6);
  }
  if ( a5 )
  {
    v20[0] = sub_16498A0(v10);
    if ( a5 == 1 )
      v18 = sub_161BE60(v20, 2000, 1);
    else
      v18 = sub_161BE60(v20, 1, 2000);
    sub_1625C10(v10, 2, v18);
  }
  return v10;
}
