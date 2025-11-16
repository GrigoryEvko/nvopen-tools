// Function: sub_EAD1B0
// Address: 0xead1b0
//
__int64 __fastcall sub_EAD1B0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rdi
  __int64 v4; // rsi
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // [rsp+18h] [rbp-48h] BYREF
  const char *v7; // [rsp+20h] [rbp-40h] BYREF
  char v8; // [rsp+40h] [rbp-20h]
  char v9; // [rsp+41h] [rbp-1Fh]

  v1 = sub_ECD690(a1 + 40);
  if ( !*(_BYTE *)(a1 + 869) && (unsigned __int8)sub_EA2540(a1) )
    return 1;
  if ( (unsigned __int8)sub_EAC8B0(a1, &v6) )
    return 1;
  if ( (unsigned __int8)sub_ECE000(a1) )
    return 1;
  v9 = 1;
  v7 = "invalid bundle alignment size (expected between 0 and 30)";
  v8 = 3;
  if ( (unsigned __int8)sub_ECE070(a1, v6 > 0x1E, v1, &v7) )
    return 1;
  v3 = *(_QWORD *)(a1 + 232);
  v4 = 0xFFFFFFFFLL;
  if ( 1LL << v6 )
  {
    _BitScanReverse64(&v5, 1LL << v6);
    v4 = 63 - ((unsigned int)v5 ^ 0x3F);
  }
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v3 + 1240LL))(v3, v4);
  return 0;
}
