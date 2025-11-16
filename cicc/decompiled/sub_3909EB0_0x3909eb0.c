// Function: sub_3909EB0
// Address: 0x3909eb0
//
__int64 __fastcall sub_3909EB0(unsigned int *a1, int a2)
{
  unsigned int v2; // r13d
  int v4; // esi
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+10h] [rbp-30h]
  char v8; // [rsp+11h] [rbp-2Fh]

  v4 = *(_DWORD *)sub_3909460((__int64)a1);
  if ( v4 == a2 )
  {
    v8 = 1;
    v6 = "unexpected token";
    v7 = 3;
    sub_3909E20(a1, v4, (__int64)&v6);
  }
  LOBYTE(v2) = v4 == a2;
  return v2;
}
