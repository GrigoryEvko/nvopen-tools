// Function: sub_62B780
// Address: 0x62b780
//
__int64 __fastcall sub_62B780(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v8; // [rsp+8h] [rbp-148h] BYREF
  _BYTE v9[64]; // [rsp+10h] [rbp-140h] BYREF
  _BYTE v10[96]; // [rsp+50h] [rbp-100h] BYREF
  char v11[8]; // [rsp+B0h] [rbp-A0h] BYREF
  _BYTE v12[57]; // [rsp+B8h] [rbp-98h] BYREF
  char v13; // [rsp+F1h] [rbp-5Fh]

  v8 = sub_72CBE0();
  memset(v10, 0, 0x58u);
  sub_87A720(42, v9, &dword_4F063F8);
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  sub_7B8B50(42, v9, v1, v2);
  sub_87E3B0(v11);
  v13 |= 8u;
  sub_627530(a1, 0, &v8, v11, v9, 0, 0, 0, 0, 0, 0, 1, 1, (__int64)v10);
  if ( dword_4F04C64 == -1
    || (v3 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v3 + 7) & 1) == 0)
    || dword_4F04C44 == -1 && (*(_BYTE *)(v3 + 6) & 2) == 0 )
  {
    if ( (v13 & 8) == 0 )
      sub_87E280(v12);
  }
  v4 = v8;
  if ( v8 )
  {
    if ( *(_BYTE *)(v8 + 140) == 7 )
    {
      v5 = *(__int64 **)(v8 + 168);
      v4 = *v5;
      if ( *v5 )
      {
        v6 = *v5;
        do
        {
          *(_BYTE *)(v6 + 34) |= 0x80u;
          v6 = *(_QWORD *)v6;
        }
        while ( v6 );
      }
    }
    else
    {
      v4 = 0;
    }
  }
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  return v4;
}
