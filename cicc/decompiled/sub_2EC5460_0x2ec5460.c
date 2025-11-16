// Function: sub_2EC5460
// Address: 0x2ec5460
//
__int64 __fastcall sub_2EC5460(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // r12

  *a1 = &unk_4A29A70;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  result = sub_22077B0(0x140u);
  v2 = result;
  if ( result )
    result = sub_2F5FEE0(result);
  a1[7] = v2;
  return result;
}
