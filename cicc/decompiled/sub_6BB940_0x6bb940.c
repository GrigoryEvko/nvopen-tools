// Function: sub_6BB940
// Address: 0x6bb940
//
__int64 __fastcall sub_6BB940(__int64 a1, int a2, int a3)
{
  char *v4; // rcx
  __int64 v5; // rax
  __int64 v6; // r13
  char v8; // [rsp+4h] [rbp-CCh] BYREF
  __int64 v9; // [rsp+8h] [rbp-C8h] BYREF
  char v10[192]; // [rsp+10h] [rbp-C0h] BYREF

  sub_6E2250(v10, &v9, 4, 1, a1, 0);
  v4 = &v8;
  *(_BYTE *)(a1 + 129) = *(_BYTE *)(a1 + 129) & 0xF9 | (2 * (a2 & 1)) | 4;
  if ( !a3 )
    v4 = 0;
  v5 = sub_6BB770(a1, 1u, a2, v4);
  v6 = v5;
  if ( v5 )
  {
    if ( !a2 )
      sub_6E65B0(v5);
    if ( !*(_BYTE *)(v6 + 8) && (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x20) != 0 )
      *(_BYTE *)(v6 + 9) |= 0x40u;
  }
  sub_6E2C70(v9, 1, a1, 0);
  return v6;
}
