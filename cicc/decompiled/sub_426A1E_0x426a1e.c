// Function: sub_426A1E
// Address: 0x426a1e
//
void __fastcall __noreturn sub_426A1E(__int64 a1)
{
  _QWORD *v1; // rbp
  __int64 v2; // rax
  int v3; // [rsp+0h] [rbp-28h] BYREF
  __int64 v4; // [rsp+8h] [rbp-20h]

  v1 = (_QWORD *)sub_2252770(48);
  v3 = 1;
  v4 = sub_2257530();
  sub_22575B0(v1, a1, &v3);
  *v1 = off_4A081F0;
  v2 = sub_2255C20(v1);
  sub_2257740(v1 + 4, v2);
  sub_2253480(v1, &`typeinfo for'std::__ios_failure, sub_2257460);
}
