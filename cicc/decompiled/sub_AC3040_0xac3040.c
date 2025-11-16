// Function: sub_AC3040
// Address: 0xac3040
//
__int64 __fastcall sub_AC3040(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rdi

  sub_BD35F0(a1, a2, 18);
  *(_DWORD *)(a1 + 4) &= 0x38000000u;
  v7 = sub_C33340(a1, a2, v4, v5, v6);
  v8 = a1 + 24;
  if ( *a3 == v7 )
    return sub_C3C790(v8, a3);
  else
    return sub_C33EB0(v8, a3);
}
