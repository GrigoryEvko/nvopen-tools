// Function: sub_691730
// Address: 0x691730
//
__int64 __fastcall sub_691730(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 result; // rax

  v2 = a2[11];
  v3 = sub_6F6F40(a2, 0);
  v4 = sub_691700(v3, a1, 0);
  result = sub_6E2DD0(a2, 1);
  a2[18] = v4;
  *a2 = a1;
  a2[11] = v2;
  return result;
}
