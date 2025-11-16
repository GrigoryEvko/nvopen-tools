// Function: sub_12A8F50
// Address: 0x12a8f50
//
_QWORD *__fastcall sub_12A8F50(__int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v16; // [rsp+8h] [rbp-58h] BYREF
  char v17[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v18; // [rsp+20h] [rbp-40h]

  v18 = 257;
  v7 = sub_1648A60(64, 2);
  v8 = (_QWORD *)v7;
  if ( v7 )
    sub_15F9650(v7, a2, a3, a4, 0);
  v9 = a1[1];
  if ( v9 )
  {
    v10 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v9 + 40, v8);
    v11 = v8[3];
    v12 = *v10;
    v8[4] = v10;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    v8[3] = v12 | v11 & 7;
    *(_QWORD *)(v12 + 8) = v8 + 3;
    *v10 = *v10 & 7 | (unsigned __int64)(v8 + 3);
  }
  sub_164B780(v8, v17);
  v13 = *a1;
  if ( *a1 )
  {
    v16 = *a1;
    sub_1623A60(&v16, v13, 2);
    if ( v8[6] )
      sub_161E7C0(v8 + 6);
    v14 = v16;
    v8[6] = v16;
    if ( v14 )
      sub_1623210(&v16, v14, v8 + 6);
  }
  return v8;
}
