// Function: sub_15E6530
// Address: 0x15e6530
//
bool __fastcall sub_15E6530(__int64 a1)
{
  __int64 v1; // rbp
  bool result; // al
  char v3; // al
  __int64 v4; // rdx
  char v5; // dl
  _QWORD *v6; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v7[4]; // [rsp-48h] [rbp-48h] BYREF
  int v8; // [rsp-24h] [rbp-24h]
  __int64 v9; // [rsp-8h] [rbp-8h]

  if ( (*(_BYTE *)(a1 + 32) & 0xF) == 1 )
    return 0;
  v9 = v1;
  if ( sub_15E4F60(a1) )
    return 0;
  v3 = *(_BYTE *)(a1 + 32) & 0xF;
  if ( ((v3 + 14) & 0xFu) <= 3 )
    return 0;
  if ( ((v3 + 7) & 0xFu) <= 1 )
    return 0;
  sub_15E64D0(a1);
  if ( v4 )
  {
    if ( (unsigned int)sub_15E4C60(a1) )
      return 0;
  }
  if ( *(_QWORD *)(a1 + 40) )
  {
    sub_16E1010(&v6);
    if ( v8 != 2 )
    {
      if ( v6 != v7 )
        j_j___libc_free_0(v6, v7[0] + 1LL);
      return 1;
    }
    if ( v6 != v7 )
      j_j___libc_free_0(v6, v7[0] + 1LL);
  }
  v5 = *(_BYTE *)(a1 + 32);
  result = 1;
  if ( (v5 & 0x30) == 0 )
    return (v5 & 0xFu) - 7 <= 1;
  return result;
}
