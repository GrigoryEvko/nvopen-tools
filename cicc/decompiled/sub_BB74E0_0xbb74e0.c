// Function: sub_BB74E0
// Address: 0xbb74e0
//
__int64 __fastcall sub_BB74E0(__int64 *a1, unsigned int *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  bool v4; // zf
  int v6; // [rsp+Ch] [rbp-4h] BYREF

  v2 = *a1;
  v3 = *a2;
  v4 = *(_QWORD *)(*a1 + 16) == 0;
  v6 = *a2;
  if ( v4 )
    sub_4263D6(a1, a2, v3);
  return (*(__int64 (__fastcall **)(__int64, int *))(v2 + 24))(v2, &v6);
}
