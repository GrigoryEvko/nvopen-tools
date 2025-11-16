// Function: sub_2C95850
// Address: 0x2c95850
//
__int64 __fastcall sub_2C95850(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v10; // rsi
  __int64 v11; // rsi
  int v12; // [rsp+8h] [rbp-E8h]
  const char *v13; // [rsp+10h] [rbp-E0h] BYREF
  char v14; // [rsp+30h] [rbp-C0h]
  char v15; // [rsp+31h] [rbp-BFh]
  __int64 v16[2]; // [rsp+40h] [rbp-B0h] BYREF
  char v17; // [rsp+50h] [rbp-A0h] BYREF
  void *v18; // [rsp+C0h] [rbp-30h]

  v5 = a2;
  if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8) )
    return sub_BD2ED0(a3, a1, v5);
  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    v7 = a3 + 24;
    goto LABEL_7;
  }
  v6 = *(_QWORD *)(a2 + 40);
  if ( *(_BYTE *)a2 != 84 )
  {
    v10 = *(_QWORD *)(a2 + 32);
    if ( v10 == v6 + 48 || !v10 )
      v11 = 0;
    else
      v11 = v10 - 24;
    v7 = v11 + 24;
    goto LABEL_7;
  }
  v7 = sub_AA4FF0(v6);
  if ( v7 )
LABEL_7:
    v7 -= 24;
  sub_23D0AB0((__int64)v16, v7, 0, 0, 0);
  v8 = *(_QWORD *)(a1 + 8);
  v13 = "bitCast";
  v15 = 1;
  v14 = 3;
  v5 = sub_2C91010(v16, 49, v5, v8, (__int64)&v13, 0, v12, 0);
  nullsub_61();
  v18 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v16[0] != &v17 )
    _libc_free(v16[0]);
  return sub_BD2ED0(a3, a1, v5);
}
