// Function: sub_7E4A00
// Address: 0x7e4a00
//
_BOOL8 __fastcall sub_7E4A00(__int64 a1)
{
  unsigned int v2; // r13d
  int v3; // eax
  int v4; // [rsp+0h] [rbp-30h] BYREF
  int v5; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( !sub_7E16F0() || (unsigned int)sub_736DD0(a1) )
    return 0;
  v2 = dword_4D03F94;
  dword_4D03F94 = 1;
  sub_7E3EE0(a1);
  v3 = sub_7E3BF0(a1, &v4, v6, &v5);
  dword_4D03F94 = v2;
  return v3 != 0;
}
