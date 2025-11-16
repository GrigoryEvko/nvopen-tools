// Function: sub_1277140
// Address: 0x1277140
//
__int64 __fastcall sub_1277140(__int64 *a1, __int64 a2, __int64 a3)
{
  char *v5; // r14
  unsigned int v6; // eax
  __int64 v7; // rdi
  int v8; // r12d
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // r15
  char v12; // al
  __int64 v13; // rdx
  char *v14; // rax
  __int64 v16; // [rsp+8h] [rbp-38h]

  v5 = (char *)sub_127B360(a2);
  if ( !a3 )
    a3 = sub_127A040(a1 + 1, *(_QWORD *)(a2 + 120));
  v6 = sub_127BFC0(a2);
  v7 = a3;
  v8 = 7;
  v9 = sub_1646BA0(v7, v6);
  v10 = *(_BYTE *)(a2 + 136);
  v11 = v9;
  v12 = *(_BYTE *)(a2 + 156);
  if ( v10 != 2 )
  {
    if ( v12 < 0 || ((v12 & 0x40) != 0 || (*(_QWORD *)(a2 + 168) & 0x2000100000LL) != 0) && *(_QWORD *)(a2 + 240) )
    {
      v8 = 3;
      if ( (*(_BYTE *)(a2 + 174) & 1) == 0 )
        v8 = (*(_BYTE *)(a2 + 176) & 0x20) == 0 ? 5 : 3;
    }
    else
    {
      if ( v10 > 1u )
        sub_127B550("unsupported storage class!");
      v8 = 0;
    }
  }
  v13 = qword_4F04C50;
  if ( qword_4F04C50 )
    v13 = *(_QWORD *)(qword_4F04C50 + 32LL);
  v16 = v13;
  if ( (v12 & 1) != 0 || (unsigned __int8)sub_127BF90(*(_QWORD *)(a2 + 120)) || (*(_BYTE *)(a2 + 176) & 0x20) != 0 )
    return sub_12769C0(a1, v5, v11, v8, a2);
  if ( !v16 || (*(_BYTE *)(v16 + 197) & 4) == 0 )
  {
    if ( v5 )
    {
      v14 = sub_693CD0(v5);
      sub_6851A0(0xDC5u, dword_4F07508, (__int64)v14);
    }
    else
    {
      sub_6851C0(0xDC6u, dword_4F07508);
    }
  }
  return sub_15A06D0(v11);
}
