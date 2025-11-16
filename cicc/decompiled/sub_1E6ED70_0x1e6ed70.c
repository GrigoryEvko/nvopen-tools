// Function: sub_1E6ED70
// Address: 0x1e6ed70
//
__int64 __fastcall sub_1E6ED70(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // r12

  *a1 = &unk_49FC458;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  result = sub_22077B0(96);
  v2 = result;
  if ( result )
    result = sub_1ED72C0(result);
  a1[7] = v2;
  return result;
}
