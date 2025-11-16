// Function: sub_3881F70
// Address: 0x3881f70
//
unsigned __int64 __fastcall sub_3881F70(__int64 a1, char *a2, char *a3)
{
  __int64 v3; // rbp
  unsigned __int64 i; // r8
  unsigned __int64 v6; // rsi
  const char *v7; // [rsp-28h] [rbp-28h] BYREF
  char v8; // [rsp-18h] [rbp-18h]
  char v9; // [rsp-17h] [rbp-17h]
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( a2 == a3 )
    return 0;
  for ( i = *a2 - 48; a3 != ++a2; i = *a2 - 48 + 10 * i )
  {
    if ( *a2 - 48 + 10 * i < i )
    {
      v10 = v3;
      v6 = *(_QWORD *)(a1 + 48);
      v9 = 1;
      v7 = "constant bigger than 64 bits detected!";
      v8 = 3;
      sub_38814C0(a1, v6, (__int64)&v7);
      return 0;
    }
  }
  return i;
}
