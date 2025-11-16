// Function: sub_1CB79F0
// Address: 0x1cb79f0
//
__int64 __fastcall sub_1CB79F0(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r15
  unsigned int v3; // r14d
  unsigned int v4; // eax
  int v5; // r14d
  int v6; // r8d
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 - 48);
  v3 = sub_1CB76C0(a1, *(_QWORD *)(a2 - 24));
  v4 = sub_1CB76C0(a1, v2);
  v5 = sub_1CB71C0((__int64)a1, v4, v3);
  v6 = sub_1CB76C0(a1, a2);
  result = 0;
  if ( v6 != v5 )
  {
    sub_1CB7560(a1, a2, v5);
    return 1;
  }
  return result;
}
