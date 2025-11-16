// Function: sub_2F229E0
// Address: 0x2f229e0
//
__int64 __fastcall sub_2F229E0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // r8

  v2 = a1[1];
  v3 = a1[2];
  a1[1] = 0;
  result = a1[3];
  v5 = *a1;
  a1[3] = 0;
  a1[2] = 0;
  *a1 = *a2;
  a1[1] = a2[1];
  a1[2] = a2[2];
  a1[3] = a2[3];
  *a2 = v5;
  a2[1] = v2;
  a2[2] = v3;
  a2[3] = result;
  return result;
}
