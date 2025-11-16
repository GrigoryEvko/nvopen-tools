// Function: sub_26C07A0
// Address: 0x26c07a0
//
__int64 __fastcall sub_26C07A0(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rdx
  __int64 v3; // r15
  __int128 v4; // rax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v6[0] = sub_B2D7E0(a1, "sample-profile-suffix-elision-policy", 0x24u);
  v1 = sub_A72240(v6);
  v3 = v2;
  *(_QWORD *)&v4 = sub_BD5D20(a1);
  return sub_C16140(v4, v1, v3);
}
