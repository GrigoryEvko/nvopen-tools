// Function: sub_6866A0
// Address: 0x6866a0
//
void __fastcall __noreturn sub_6866A0(int a1, int a2)
{
  char *v2; // rax
  char *v3; // rax
  __int64 v4; // r13
  char *v5; // rax

  if ( !a2 )
  {
    v2 = sub_67C860(a1);
    sub_685200(0xBDu, dword_4F07508, (__int64)v2);
  }
  v3 = strerror(a2);
  v4 = sub_724840((unsigned int)dword_4D03A00, v3);
  v5 = sub_67C860(a1);
  sub_686630(0x6A7u, (__int64)v5, v4, dword_4F07508);
}
