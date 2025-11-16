// Function: sub_1B0E2C0
// Address: 0x1b0e2c0
//
__int64 __fastcall sub_1B0E2C0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, double a5, double a6, double a7)
{
  _QWORD *v8; // r12
  __int64 v10; // r13
  unsigned __int8 v11; // al
  unsigned int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned __int64 *v16; // r13
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  unsigned __int8 *v22; // [rsp+8h] [rbp-58h] BYREF
  char v23[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v24; // [rsp+20h] [rbp-40h]

  v8 = (_QWORD *)a2;
  v10 = sub_15A0680(*(_QWORD *)a2, a3, 0);
  v11 = *(_BYTE *)(v10 + 16);
  if ( v11 > 0x10u )
  {
LABEL_8:
    v24 = 257;
    v14 = sub_15FB440(26, (__int64 *)a2, v10, (__int64)v23, 0);
    v15 = a1[1];
    v8 = (_QWORD *)v14;
    if ( v15 )
    {
      v16 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v15 + 40, v14);
      v17 = v8[3];
      v18 = *v16;
      v8[4] = v16;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      v8[3] = v18 | v17 & 7;
      *(_QWORD *)(v18 + 8) = v8 + 3;
      *v16 = *v16 & 7 | (unsigned __int64)(v8 + 3);
    }
    sub_164B780((__int64)v8, a4);
    v19 = *a1;
    if ( *a1 )
    {
      v22 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v22, v19, 2);
      v20 = v8[6];
      if ( v20 )
        sub_161E7C0((__int64)(v8 + 6), v20);
      v21 = v22;
      v8[6] = v22;
      if ( v21 )
        sub_1623210((__int64)&v22, v21, (__int64)(v8 + 6));
    }
    return (__int64)v8;
  }
  if ( v11 != 13 )
  {
LABEL_3:
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
      return sub_15A2CF0((__int64 *)a2, v10, a5, a6, a7);
    goto LABEL_8;
  }
  v13 = *(_DWORD *)(v10 + 32);
  if ( v13 <= 0x40 )
  {
    if ( *(_QWORD *)(v10 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) )
      goto LABEL_3;
  }
  else if ( v13 != (unsigned int)sub_16A58F0(v10 + 24) )
  {
    goto LABEL_3;
  }
  return (__int64)v8;
}
