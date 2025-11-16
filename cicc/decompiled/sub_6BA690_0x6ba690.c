// Function: sub_6BA690
// Address: 0x6ba690
//
__int64 __fastcall sub_6BA690(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al
  unsigned int v7; // r12d
  int v9; // [rsp+Ch] [rbp-24h] BYREF
  __int64 v10; // [rsp+10h] [rbp-20h] BYREF
  _QWORD v11[3]; // [rsp+18h] [rbp-18h] BYREF

  v10 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v11[0] = *(_QWORD *)&dword_4F063F8;
  sub_6BA680(v10);
  v6 = *(_BYTE *)(v10 + 173);
  if ( v6 == 1 )
  {
    if ( (int)sub_6210B0(v10, 0) < 0 || (v7 = 1, *a1 = sub_620FD0(v10, &v9), v9) )
    {
      v7 = 0;
      sub_6851C0(0xAFu, v11);
    }
  }
  else if ( v6 == 12 )
  {
    v7 = 0;
    sub_6851C0(0x772u, v11);
  }
  else
  {
    if ( v6 )
      sub_721090(v10);
    v7 = 0;
  }
  sub_724E30(&v10);
  return v7;
}
