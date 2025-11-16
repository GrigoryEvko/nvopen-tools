// Function: sub_426AAD
// Address: 0x426aad
//
void __fastcall __noreturn sub_426AAD(__int64 a1, int a2)
{
  _QWORD *v2; // rbp
  __int64 v3; // rax
  int v4; // [rsp+0h] [rbp-28h] BYREF
  __int64 v5; // [rsp+8h] [rbp-20h]

  v2 = (_QWORD *)sub_2252770(48);
  if ( a2 )
  {
    v4 = a2;
    v5 = sub_2241E50();
  }
  else
  {
    v4 = 1;
    v5 = sub_2257530();
  }
  sub_22575B0(v2, a1, &v4);
  *v2 = off_4A081F0;
  v3 = sub_2255C20(v2);
  sub_2257740(v2 + 4, v3);
  sub_2253480(v2, &`typeinfo for'std::__ios_failure, sub_2257460);
}
