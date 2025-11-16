// Function: sub_2461FD0
// Address: 0x2461fd0
//
__int64 __fastcall sub_2461FD0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx

  v3 = *a1;
  if ( !byte_4FE7DC8 )
    return sub_2A3ED40(v3, a2, 0, 0);
  v4 = sub_BAA410(v3, "msan.module_ctor", 0x10u);
  sub_B2F990(a2, v4, v5, v6);
  return sub_2A3ED40(*a1, a2, 0, a2);
}
