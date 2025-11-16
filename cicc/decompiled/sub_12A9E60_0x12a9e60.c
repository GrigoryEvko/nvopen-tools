// Function: sub_12A9E60
// Address: 0x12a9e60
//
__int64 __fastcall sub_12A9E60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+28h] [rbp-58h] BYREF
  char v26[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v27; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A3AE0(a2, a3, a4, 0);
  v27 = 257;
  v9 = sub_1648A60(88, 1);
  v10 = (_QWORD *)v9;
  if ( v9 )
  {
    v11 = a4;
    v22 = a4;
    v24 = v9;
    v12 = sub_15FB2A0(*(_QWORD *)a2, a3, v11);
    sub_15F1EA0(v10, v12, 62, v10 - 3, 1, 0);
    if ( *(v10 - 3) )
    {
      v13 = *(v10 - 2);
      v14 = *(v10 - 1) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v14 = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
    }
    *(v10 - 3) = a2;
    v15 = *(_QWORD *)(a2 + 8);
    *(v10 - 2) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = (unsigned __int64)(v10 - 2) | *(_QWORD *)(v15 + 16) & 3LL;
    *(v10 - 1) = (a2 + 8) | *(v10 - 1) & 3LL;
    *(_QWORD *)(a2 + 8) = v10 - 3;
    v10[7] = v10 + 9;
    v10[8] = 0x400000000LL;
    sub_15FB110(v10, a3, v22, v26);
  }
  else
  {
    v24 = 0;
  }
  v16 = a1[1];
  if ( v16 )
  {
    v17 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v16 + 40, v10);
    v18 = v10[3];
    v19 = *v17;
    v10[4] = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v10 + 3;
    *v17 = *v17 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v24, a5);
  v20 = *a1;
  if ( *a1 )
  {
    v25 = *a1;
    sub_1623A60(&v25, v20, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v21 = v25;
    v10[6] = v25;
    if ( v21 )
      sub_1623210(&v25, v21, v10 + 6);
  }
  return (__int64)v10;
}
