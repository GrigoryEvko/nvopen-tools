// Function: sub_24754F0
// Address: 0x24754f0
//
void __fastcall sub_24754F0(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // rbx
  _QWORD *v6; // rcx
  unsigned __int64 v7; // rax
  int v8; // edx
  int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 **v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  unsigned __int64 v18; // [rsp+8h] [rbp-128h]
  __int64 v19; // [rsp+18h] [rbp-118h]
  _QWORD v20[2]; // [rsp+30h] [rbp-100h] BYREF
  _BYTE v21[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v22; // [rsp+60h] [rbp-D0h]
  unsigned int *v23[24]; // [rsp+70h] [rbp-C0h] BYREF

  sub_23D0AB0((__int64)v23, a2, 0, 0, 0);
  v3 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = *(_QWORD *)(a2 + v4);
  if ( (_BYTE)qword_4FE84C8 )
    sub_2472230(a1, *(_QWORD *)(a2 + v4), a2);
  v6 = sub_2463540((__int64 *)a1, *(_QWORD *)(v3 + 8));
  if ( **(_BYTE **)(a1 + 8) )
    v7 = (unsigned __int64)sub_2465B30((__int64 *)a1, v3, (__int64)v23, (__int64)v6, 0);
  else
    v7 = sub_2463FC0(a1, v3, v23, 0x100u);
  v9 = v8;
  v20[0] = v7;
  v10 = *(_QWORD *)(a2 - 32);
  v20[1] = v5;
  v22 = 257;
  HIDWORD(v19) = 0;
  if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v11 = sub_B35180((__int64)v23, *(_QWORD *)(a2 + 8), *(_DWORD *)(v10 + 36), (__int64)v20, 2u, v19, (__int64)v21);
  v12 = *(_QWORD *)(a2 + 8);
  v22 = 257;
  v18 = v11;
  v13 = (__int64 **)sub_2463540((__int64 *)a1, v12);
  v14 = sub_24633A0((__int64 *)v23, 0x31u, v18, v13, (__int64)v21, 0, v19, 0);
  sub_246EF60(a1, a2, v14);
  v15 = *(_QWORD *)(a1 + 8);
  v16 = *(unsigned int *)(v15 + 4);
  if ( (_DWORD)v16 )
  {
    v22 = 257;
    v17 = sub_A82CA0(v23, *(_QWORD *)(v15 + 88), v9, 0, 0, (__int64)v21);
    v16 = a2;
    sub_246F1C0(a1, a2, v17);
  }
  sub_F94A20(v23, v16);
}
