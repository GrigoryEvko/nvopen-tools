// Function: sub_685D60
// Address: 0x685d60
//
void __fastcall __noreturn sub_685D60(int a1, int a2, __int64 a3, unsigned __int8 a4)
{
  int v6[14]; // [rsp+8h] [rbp-38h] BYREF

  sub_720D70(v6);
  if ( a1 )
    v6[0] |= 0x10u;
  if ( a4 == 10 )
  {
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 1;
  }
  sub_685AD0(a4, a2, a3, v6);
  sub_720FF0(11);
}
