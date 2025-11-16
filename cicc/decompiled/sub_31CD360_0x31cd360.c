// Function: sub_31CD360
// Address: 0x31cd360
//
char __fastcall sub_31CD360(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // r15
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // r13
  __int64 v14; // [rsp+8h] [rbp-88h]
  __int64 v15[4]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v16[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v17; // [rsp+50h] [rbp-40h]

  LOBYTE(v3) = sub_B46500((unsigned __int8 *)a2);
  if ( !(_BYTE)v3 && (*(_BYTE *)(a2 + 2) & 1) == 0 )
  {
    v3 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    if ( *(_DWORD *)(v3 + 8) <= 0x1FFu )
    {
      v4 = (__int64 *)sub_B43CA0(a2);
      v14 = *(_QWORD *)(a2 - 32);
      v16[0] = *(_QWORD *)(a2 + 8);
      v16[1] = *(_QWORD *)(v14 + 8);
      v5 = sub_B6E160(v4, 0x22D8u, (__int64)v16, 2);
      v6 = (_QWORD *)*v4;
      v7 = 0;
      v8 = v5;
      v9 = sub_BCB2E0(v6);
      v10 = sub_ACD640(v9, 0, 0);
      v17 = 257;
      v11 = *(_QWORD *)(a1 + 304);
      v15[0] = v10;
      v15[1] = v14;
      v15[2] = *(_QWORD *)(v11 + 32 * (1LL - (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
      if ( v8 )
        v7 = *(_QWORD *)(v8 + 24);
      v12 = sub_BD2C40(88, 4u);
      if ( v12 )
      {
        sub_B44260((__int64)v12, **(_QWORD **)(v7 + 16), 56, 4u, a2 + 24, 0);
        v12[9] = 0;
        sub_B4A290((__int64)v12, v7, v8, v15, 3, (__int64)v16, 0, 0);
      }
      sub_BD84D0(a2, (__int64)v12);
      LOBYTE(v3) = sub_B43D60((_QWORD *)a2);
    }
  }
  return v3;
}
