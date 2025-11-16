// Function: sub_692200
// Address: 0x692200
//
__int64 __fastcall sub_692200(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  _BYTE v13[4]; // [rsp+10h] [rbp-5B0h] BYREF
  int v14; // [rsp+14h] [rbp-5ACh] BYREF
  __int64 v15; // [rsp+18h] [rbp-5A8h] BYREF
  __int64 v16; // [rsp+20h] [rbp-5A0h] BYREF
  __int64 *v17; // [rsp+28h] [rbp-598h] BYREF
  _BYTE v18[160]; // [rsp+30h] [rbp-590h] BYREF
  _QWORD v19[19]; // [rsp+D0h] [rbp-4F0h] BYREF
  __int64 v20; // [rsp+168h] [rbp-458h] BYREF
  _BYTE v21[352]; // [rsp+170h] [rbp-450h] BYREF
  _BYTE v22[352]; // [rsp+2D0h] [rbp-2F0h] BYREF
  _BYTE v23[400]; // [rsp+430h] [rbp-190h] BYREF

  v17 = 0;
  v2 = qword_4D03C58;
  sub_6E1DD0(&v16);
  sub_6E1E00(5, v18, 0, 1);
  *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x11080u;
  qword_4D03C58 = &v17;
  memset(v19, 0, sizeof(v19));
  v15 = sub_724DC0(&v20, v18, v3, 0, v4, v5);
  v6 = sub_73C570(a1, 1, -1);
  v7 = sub_72D2E0(v6, 0);
  sub_72BB40(v7, v15);
  v8 = sub_730690(v15);
  v9 = sub_73DCD0(v8);
  sub_6E7150(v9, v21);
  v10 = sub_730690(v15);
  v11 = sub_73DCD0(v10);
  sub_6E7150(v11, v22);
  sub_84EC30(
    *(unsigned __int8 *)(a2 + 176),
    0,
    0,
    1,
    0,
    (unsigned int)v21,
    (__int64)v22,
    (__int64)dword_4F07508,
    dword_4F06650[0],
    0,
    0,
    (__int64)v23,
    (__int64)v13,
    (__int64)v19,
    (__int64)&v14);
  if ( v14 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 && v19[1] )
  {
    sub_724E30(&v15);
    if ( v17 )
    {
      if ( (*(_BYTE *)(a2 + 193) & 0x12) == 2 )
      {
        if ( (*(_BYTE *)(a2 + 195) & 3) != 1 && ((*(_BYTE *)(a2 + 206) & 8) == 0 || (*(_BYTE *)(a2 + 193) & 1) != 0) )
          sub_6854C0(0xC1Cu, (FILE *)(a2 + 64), *v17);
        *(_BYTE *)(a2 + 193) &= ~2u;
      }
    }
    else
    {
      *(_BYTE *)(a2 + 193) |= 2u;
    }
  }
  else
  {
    sub_724E30(&v15);
    *(_BYTE *)(a2 + 193) |= 0x22u;
    *(_BYTE *)(a2 + 206) |= 0x10u;
  }
  qword_4D03C58 = v2;
  sub_6E2B30();
  return sub_6E1DF0(v16);
}
