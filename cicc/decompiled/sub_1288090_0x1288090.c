// Function: sub_1288090
// Address: 0x1288090
//
__int64 __fastcall sub_1288090(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 *v4; // rbx
  _QWORD *v5; // r12
  unsigned int v7; // r14d
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 *v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+18h] [rbp-68h] BYREF
  char *v18; // [rsp+20h] [rbp-60h] BYREF
  char v19; // [rsp+30h] [rbp-50h]
  char v20; // [rsp+31h] [rbp-4Fh]
  char v21[16]; // [rsp+40h] [rbp-40h] BYREF
  __int16 v22; // [rsp+50h] [rbp-30h]

  v18 = "and";
  v3 = *(_BYTE *)(a3 + 16);
  v20 = 1;
  v4 = *(__int64 **)(a1 + 8);
  v19 = 3;
  if ( v3 > 0x10u )
    goto LABEL_9;
  if ( v3 != 13 )
    goto LABEL_3;
  v7 = *(_DWORD *)(a3 + 32);
  if ( v7 <= 0x40 )
  {
    v5 = (_QWORD *)a2;
    if ( *(_QWORD *)(a3 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) )
      return (__int64)v5;
LABEL_3:
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
      return sub_15A2CF0(a2, a3);
    goto LABEL_9;
  }
  v16 = a3;
  v5 = (_QWORD *)a2;
  v8 = sub_16A58F0(a3 + 24);
  a3 = v16;
  if ( v7 == v8 )
    return (__int64)v5;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A2CF0(a2, a3);
LABEL_9:
  v22 = 257;
  v9 = sub_15FB440(26, a2, a3, v21, 0);
  v10 = v4[1];
  v5 = (_QWORD *)v9;
  if ( v10 )
  {
    v11 = (unsigned __int64 *)v4[2];
    sub_157E9D0(v10 + 40, v9);
    v12 = v5[3];
    v13 = *v11;
    v5[4] = v11;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    v5[3] = v13 | v12 & 7;
    *(_QWORD *)(v13 + 8) = v5 + 3;
    *v11 = *v11 & 7 | (unsigned __int64)(v5 + 3);
  }
  sub_164B780(v5, &v18);
  v14 = *v4;
  if ( !*v4 )
    return (__int64)v5;
  v17 = *v4;
  sub_1623A60(&v17, v14, 2);
  if ( v5[6] )
    sub_161E7C0(v5 + 6);
  v15 = v17;
  v5[6] = v17;
  if ( !v15 )
    return (__int64)v5;
  sub_1623210(&v17, v15, v5 + 6);
  return (__int64)v5;
}
