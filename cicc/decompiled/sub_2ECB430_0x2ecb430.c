// Function: sub_2ECB430
// Address: 0x2ecb430
//
__int64 __fastcall sub_2ECB430(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax

  a1[1] = a2;
  sub_2ECB100(a2);
  v2 = a1[1];
  v3 = *(_QWORD *)(v2 + 3552);
  a1[3] = v2 + 3560;
  result = a1[5];
  a1[2] = v3;
  if ( a1[6] != result )
    a1[6] = result;
  return result;
}
