// Function: sub_691E30
// Address: 0x691e30
//
_QWORD *__fastcall sub_691E30(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 **i; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+10h] [rbp-520h]
  __int64 v22; // [rsp+28h] [rbp-508h] BYREF
  __int64 v23; // [rsp+30h] [rbp-500h] BYREF
  __int64 *v24; // [rsp+38h] [rbp-4F8h] BYREF
  _BYTE v25[160]; // [rsp+40h] [rbp-4F0h] BYREF
  __int64 v26[44]; // [rsp+E0h] [rbp-450h] BYREF
  __int64 v27[44]; // [rsp+240h] [rbp-2F0h] BYREF
  _QWORD v28[50]; // [rsp+3A0h] [rbp-190h] BYREF

  v24 = 0;
  v21 = qword_4D03C58;
  if ( (unsigned int)sub_8D24D0() || (*(_BYTE *)((*a1)[12] + 178LL) & 0x20) != 0 )
    goto LABEL_3;
  sub_6E1DD0(&v23);
  sub_6E1E00(5, v25, 0, 1);
  *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x11080u;
  qword_4D03C58 = &v24;
  v22 = sub_724DC0(5, v25, v3, &qword_4D03C58, v4, v5);
  for ( i = (__int64 **)*a1[21]; i; i = (__int64 **)*i )
  {
    if ( ((_BYTE)i[12] & 1) != 0 )
    {
      v7 = sub_73C570(i[5], 1, -1);
      v8 = sub_72D2E0(v7, 0);
      sub_72BB40(v8, v22);
      v9 = sub_730690(v22);
      v10 = sub_73DCD0(v9);
      sub_6E7150(v10, v26);
      v11 = sub_730690(v22);
      v12 = sub_73DCD0(v11);
      sub_6E7150(v12, v27);
      sub_6907F0(v26, v27, 0x2Fu, &dword_4F063F8, dword_4F06650[0], (__int64)v28);
      sub_688FA0(v28);
      sub_6E4710(v28);
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 )
        goto LABEL_18;
    }
  }
  v13 = *(_QWORD *)(*a1)[12];
  if ( v13 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(v13 + 80) == 8 )
      {
        v14 = *(_QWORD *)(*(_QWORD *)(v13 + 88) + 120LL);
        if ( (unsigned int)sub_8D3410(v14) )
          v14 = sub_8D40F0(v14);
        if ( (unsigned int)sub_8D3A70(v14) || (unsigned int)sub_8D2870(v14) )
        {
          v15 = sub_73C570(v14, 1, -1);
          v16 = sub_72D2E0(v15, 0);
          sub_72BB40(v16, v22);
          v17 = sub_730690(v22);
          v18 = sub_73DCD0(v17);
          sub_6E7150(v18, v26);
          v19 = sub_730690(v22);
          v20 = sub_73DCD0(v19);
          sub_6E7150(v20, v27);
          sub_6907F0(v26, v27, 0x2Fu, &dword_4F063F8, dword_4F06650[0], (__int64)v28);
          sub_688FA0(v28);
          sub_6E4710(v28);
          if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 )
            break;
        }
      }
      v13 = *(_QWORD *)(v13 + 16);
      if ( !v13 )
        goto LABEL_19;
    }
LABEL_18:
    sub_724E30(&v22);
    sub_6E2B30();
    sub_6E1DF0(v23);
LABEL_3:
    *(_BYTE *)(a2 + 193) |= 0x22u;
    *(_BYTE *)(a2 + 206) |= 0x10u;
    goto LABEL_4;
  }
LABEL_19:
  sub_724E30(&v22);
  sub_6E2B30();
  sub_6E1DF0(v23);
  if ( v24 )
  {
    if ( (*(_BYTE *)(a2 + 193) & 0x12) == 2 )
    {
      if ( (*(_BYTE *)(a2 + 195) & 3) != 1 && ((*(_BYTE *)(a2 + 206) & 8) == 0 || (*(_BYTE *)(a2 + 193) & 1) != 0) )
        sub_6854C0(0xC1Cu, (FILE *)(a2 + 64), *v24);
      *(_BYTE *)(a2 + 193) &= ~2u;
    }
  }
  else
  {
    *(_BYTE *)(a2 + 193) |= 2u;
  }
LABEL_4:
  qword_4D03C58 = v21;
  return &qword_4D03C58;
}
