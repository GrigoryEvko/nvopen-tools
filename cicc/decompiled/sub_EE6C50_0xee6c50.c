// Function: sub_EE6C50
// Address: 0xee6c50
//
__int64 __fastcall sub_EE6C50(__int64 *a1)
{
  __int64 v1; // rcx
  unsigned __int64 v2; // rax
  __int64 v3; // rdi
  unsigned __int64 v5; // [rsp+8h] [rbp-18h] BYREF
  __int64 v6[2]; // [rsp+10h] [rbp-10h] BYREF

  v5 = 0;
  if ( (unsigned __int8)sub_EE35F0(a1, (__int64 *)&v5) )
    return 0;
  v1 = *a1;
  v2 = v5;
  if ( a1[1] - *a1 < v5 || !v5 )
    return 0;
  v6[0] = v5;
  v3 = (__int64)(a1 + 101);
  *(_QWORD *)(v3 - 808) = v1 + v5;
  v6[1] = v1;
  if ( v2 > 9 && *(_QWORD *)v1 == 0x5F4C41424F4C475FLL && *(_WORD *)(v1 + 8) == 20063 )
    return sub_EE68C0(v3, "(anonymous namespace)");
  else
    return sub_EE6A90(v3, v6);
}
