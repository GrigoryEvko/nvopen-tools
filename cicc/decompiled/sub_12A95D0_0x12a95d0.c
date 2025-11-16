// Function: sub_12A95D0
// Address: 0x12a95d0
//
__int64 __fastcall sub_12A95D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  _QWORD *v9; // r12
  unsigned __int64 *v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // [rsp+8h] [rbp-48h] BYREF
  char v16[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v17; // [rsp+20h] [rbp-30h]

  if ( a3 == *(_QWORD *)a2 )
    return a2;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A4A70(a2, a3);
  v17 = 257;
  v7 = sub_15FDFF0(a2, a3, v16, 0);
  v8 = a1[1];
  v9 = (_QWORD *)v7;
  if ( v8 )
  {
    v10 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v8 + 40, v7);
    v11 = v9[3];
    v12 = *v10;
    v9[4] = v10;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    v9[3] = v12 | v11 & 7;
    *(_QWORD *)(v12 + 8) = v9 + 3;
    *v10 = *v10 & 7 | (unsigned __int64)(v9 + 3);
  }
  sub_164B780(v9, a4);
  v13 = *a1;
  if ( *a1 )
  {
    v15 = *a1;
    sub_1623A60(&v15, v13, 2);
    if ( v9[6] )
      sub_161E7C0(v9 + 6);
    v14 = v15;
    v9[6] = v15;
    if ( v14 )
      sub_1623210(&v15, v14, v9 + 6);
  }
  return (__int64)v9;
}
