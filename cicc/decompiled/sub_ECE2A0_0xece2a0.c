// Function: sub_ECE2A0
// Address: 0xece2a0
//
__int64 __fastcall sub_ECE2A0(__int64 a1, int a2)
{
  unsigned int v2; // r13d
  int v4; // esi
  const char *v6; // [rsp+0h] [rbp-50h] BYREF
  char v7; // [rsp+20h] [rbp-30h]
  char v8; // [rsp+21h] [rbp-2Fh]

  v4 = *(_DWORD *)sub_ECD7B0(a1);
  if ( v4 == a2 )
  {
    v8 = 1;
    v6 = "unexpected token";
    v7 = 3;
    sub_ECE210(a1, v4, (__int64)&v6);
  }
  LOBYTE(v2) = v4 == a2;
  return v2;
}
