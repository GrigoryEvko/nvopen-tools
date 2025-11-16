// Function: sub_35C5BD0
// Address: 0x35c5bd0
//
__int64 __fastcall sub_35C5BD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 result; // rax

  sub_35C5A50(a1, a2);
  result = sub_2E225E0((__int64 *)(a1 + 88), a2, v2, v3, v4, v5);
  *(_QWORD *)(a1 + 32) = a2 + 48;
  return result;
}
