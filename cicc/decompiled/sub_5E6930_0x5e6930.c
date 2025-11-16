// Function: sub_5E6930
// Address: 0x5e6930
//
__int64 __fastcall sub_5E6930(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 result; // rax
  __int64 v5; // r8
  _BYTE v6[16]; // [rsp+0h] [rbp-60h] BYREF
  char v7; // [rsp+10h] [rbp-50h]
  char v8; // [rsp+11h] [rbp-4Fh]
  char v9; // [rsp+12h] [rbp-4Eh]
  __int64 v10; // [rsp+18h] [rbp-48h]
  __int64 v11; // [rsp+20h] [rbp-40h]

  sub_878710(a1, v6);
  if ( (v8 & 0x40) == 0 )
  {
    v7 &= ~0x80u;
    v10 = 0;
  }
  v9 &= ~2u;
  v11 = 0;
  v2 = sub_7D5DD0(v6, 131080);
  v3 = v2;
  if ( !v2
    || (*(_BYTE *)(v2 + 81) & 0x10) == 0
    || (v5 = sub_8D5CE0(*(_QWORD *)(a1 + 64), *(_QWORD *)(v2 + 64)), result = 1, !v5) )
  {
    v3 = 0;
    result = 0;
  }
  *a2 = v3;
  return result;
}
