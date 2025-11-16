// Function: sub_32113D0
// Address: 0x32113d0
//
char __fastcall sub_32113D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8

  *(_QWORD *)(a1 + 56) = 0;
  LOBYTE(v2) = sub_2E31AB0(a2);
  if ( !(_BYTE)v2 )
  {
    v2 = sub_2E309C0(a2, a2, v3, v4, v5);
    *(_QWORD *)(a1 + 32) = v2;
  }
  return v2;
}
