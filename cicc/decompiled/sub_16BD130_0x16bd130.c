// Function: sub_16BD130
// Address: 0x16bd130
//
void __fastcall __noreturn sub_16BD130(_BYTE *a1, unsigned __int8 a2)
{
  bool v2; // zf
  _BYTE *v3; // [rsp+0h] [rbp-20h] BYREF
  __int16 v4; // [rsp+10h] [rbp-10h]

  v2 = *a1 == 0;
  v4 = 257;
  if ( !v2 )
  {
    v3 = a1;
    LOBYTE(v4) = 3;
  }
  sub_16BCFB0((__int64)&v3, a2);
}
