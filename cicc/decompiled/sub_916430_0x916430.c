// Function: sub_916430
// Address: 0x916430
//
__int64 __fastcall sub_916430(__int64 *a1, __int64 a2, __int64 a3)
{
  char *v5; // r15
  unsigned int v6; // r13d
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // r10
  char v11; // al
  __int64 v12; // rdx
  char v13; // al
  char *v14; // rax
  __int64 v16; // [rsp+0h] [rbp-40h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  v5 = (char *)sub_91B6B0(a2);
  if ( !a3 )
    a3 = sub_91A3A0(a1 + 1, *(_QWORD *)(a2 + 120));
  v6 = 7;
  v7 = (unsigned int)sub_91C310(a2);
  v8 = sub_BCE760(a3, v7);
  v9 = *(_BYTE *)(a2 + 136);
  v10 = v8;
  v11 = *(_BYTE *)(a2 + 156);
  if ( v9 != 2 )
  {
    if ( v11 < 0 || ((v11 & 0x40) != 0 || (*(_QWORD *)(a2 + 168) & 0x2000100000LL) != 0) && *(_QWORD *)(a2 + 240) )
    {
      v6 = 3;
      if ( (*(_BYTE *)(a2 + 174) & 1) == 0 )
        v6 = (*(_BYTE *)(a2 + 176) & 0x20) == 0 ? 5 : 3;
    }
    else
    {
      v6 = 0;
      if ( v9 > 1u )
        sub_91B8A0("unsupported storage class!");
    }
  }
  v12 = qword_4F04C50;
  if ( qword_4F04C50 )
    v12 = *(_QWORD *)(qword_4F04C50 + 32LL);
  v17 = v12;
  if ( (v11 & 1) != 0 )
    return sub_915C40(a1, v5, v10, a3, v6, a2);
  v16 = v10;
  v13 = sub_91C2E0(*(_QWORD *)(a2 + 120));
  v10 = v16;
  if ( v13 || (*(_BYTE *)(a2 + 176) & 0x20) != 0 )
    return sub_915C40(a1, v5, v10, a3, v6, a2);
  if ( v17 && (*(_BYTE *)(v17 + 197) & 4) != 0 )
    return sub_AD6530(v10);
  if ( v5 )
  {
    v14 = sub_693CD0(v5);
    sub_6851A0(0xDC5u, dword_4F07508, (__int64)v14);
    v10 = v16;
    return sub_AD6530(v10);
  }
  sub_6851C0(0xDC6u, dword_4F07508);
  return sub_AD6530(v16);
}
