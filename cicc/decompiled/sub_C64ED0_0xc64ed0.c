// Function: sub_C64ED0
// Address: 0xc64ed0
//
void __fastcall __noreturn sub_C64ED0(_BYTE *a1, unsigned __int8 a2)
{
  bool v2; // zf
  _BYTE *v3; // [rsp+0h] [rbp-30h] BYREF
  __int16 v4; // [rsp+20h] [rbp-10h]

  v2 = *a1 == 0;
  v4 = 257;
  if ( !v2 )
  {
    v3 = a1;
    LOBYTE(v4) = 3;
  }
  sub_C64D30((__int64)&v3, a2);
}
