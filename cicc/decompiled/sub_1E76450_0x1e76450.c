// Function: sub_1E76450
// Address: 0x1e76450
//
__int64 __fastcall sub_1E76450(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax

  a1[1] = a2;
  sub_1E75FC0(a2);
  v2 = a1[1];
  v3 = *(_QWORD *)(v2 + 2280);
  a1[3] = v2 + 2288;
  result = a1[5];
  a1[2] = v3;
  if ( a1[6] != result )
    a1[6] = result;
  return result;
}
