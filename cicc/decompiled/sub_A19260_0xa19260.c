// Function: sub_A19260
// Address: 0xa19260
//
__int64 __fastcall sub_A19260(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  volatile signed __int32 *v4; // r8

  v2 = *a2;
  result = a2[1];
  *a2 = 0;
  a2[1] = 0;
  v4 = (volatile signed __int32 *)a1[1];
  *a1 = v2;
  a1[1] = result;
  if ( v4 )
    return sub_A191D0(v4);
  return result;
}
