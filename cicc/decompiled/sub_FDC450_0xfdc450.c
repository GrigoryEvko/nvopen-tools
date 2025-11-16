// Function: sub_FDC450
// Address: 0xfdc450
//
__int64 __fastcall sub_FDC450(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v5; // [rsp+10h] [rbp-30h]

  v2 = *a1;
  if ( !*a1 )
    return v5;
  v3 = sub_FDC440(a1);
  return sub_FE8740(v2, v3, a2, 0);
}
