// Function: sub_1288DC0
// Address: 0x1288dc0
//
_QWORD *__fastcall sub_1288DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // al
  __int64 *v6; // rbx
  _QWORD *v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 *v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v18; // [rsp+8h] [rbp-68h] BYREF
  char *v19; // [rsp+10h] [rbp-60h] BYREF
  char v20; // [rsp+20h] [rbp-50h]
  char v21; // [rsp+21h] [rbp-4Fh]
  char v22[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v23; // [rsp+40h] [rbp-30h]

  v5 = sub_127B3A0(a4);
  v6 = *(__int64 **)(a1 + 8);
  v21 = 1;
  v20 = 3;
  v19 = "rem";
  if ( v5 )
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u )
    {
      v7 = (_QWORD *)sub_15A2A30(21, a2, a3, 0, 0);
      if ( v7 )
        return v7;
    }
    v23 = 257;
    v8 = 21;
    v9 = a3;
  }
  else
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u )
    {
      v7 = (_QWORD *)sub_15A2A30(20, a2, a3, 0, 0);
      if ( v7 )
        return v7;
    }
    v9 = a3;
    v23 = 257;
    v8 = 20;
  }
  v10 = sub_15FB440(v8, a2, v9, v22, 0);
  v11 = v6[1];
  v7 = (_QWORD *)v10;
  if ( v11 )
  {
    v12 = (unsigned __int64 *)v6[2];
    sub_157E9D0(v11 + 40, v10);
    v13 = v7[3];
    v14 = *v12;
    v7[4] = v12;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v7[3] = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v7 + 3;
    *v12 = *v12 & 7 | (unsigned __int64)(v7 + 3);
  }
  sub_164B780(v7, &v19);
  v15 = *v6;
  if ( *v6 )
  {
    v18 = *v6;
    sub_1623A60(&v18, v15, 2);
    if ( v7[6] )
      sub_161E7C0(v7 + 6);
    v16 = v18;
    v7[6] = v18;
    if ( v16 )
      sub_1623210(&v18, v16, v7 + 6);
  }
  return v7;
}
