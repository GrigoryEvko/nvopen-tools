// Function: sub_1600D80
// Address: 0x1600d80
//
__int64 __fastcall sub_1600D80(__int64 *a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r12
  char v6[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v7; // [rsp+10h] [rbp-30h]

  v1 = *(a1 - 3);
  v2 = *a1;
  v7 = 257;
  v3 = sub_1648A60(56, 1);
  v4 = v3;
  if ( v3 )
    sub_15FC990(v3, v1, v2, (__int64)v6, 0);
  return v4;
}
