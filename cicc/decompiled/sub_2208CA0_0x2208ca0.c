// Function: sub_2208CA0
// Address: 0x2208ca0
//
__int64 __fastcall sub_2208CA0(__int64 *a1)
{
  __int64 result; // rax
  __int64 *v2; // rdx

  result = *a1;
  v2 = (__int64 *)a1[1];
  *v2 = *a1;
  *(_QWORD *)(result + 8) = v2;
  return result;
}
