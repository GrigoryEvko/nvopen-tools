// Function: sub_12F9A20
// Address: 0x12f9a20
//
__int64 __fastcall sub_12F9A20(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r10
  __int64 v7; // rdi
  _BOOL8 v8; // r8
  __int64 result; // rax
  __int64 v10; // rdx

  v4 = a2[1];
  v5 = *a2;
  v6 = *a1;
  v7 = a1[1];
  v8 = v7 < 0;
  if ( v7 < 0 == v4 < 0 )
    result = sub_12FA1E0(v7, v6, v4, v5, v8);
  else
    result = sub_12F9F70(v7, v6, v4, v5, v8);
  *a3 = result;
  a3[1] = v10;
  return result;
}
