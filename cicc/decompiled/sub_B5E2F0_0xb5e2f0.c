// Function: sub_B5E2F0
// Address: 0xb5e2f0
//
__int64 __fastcall sub_B5E2F0(__int64 a1, const char *a2)
{
  size_t v2; // rdx
  __int64 v3; // rcx

  v2 = strlen(a2);
  if ( v2 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
    sub_4262D8((__int64)"basic_string::append");
  return sub_2241490(a1, a2, v2, v3);
}
