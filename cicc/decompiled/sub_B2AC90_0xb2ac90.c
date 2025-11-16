// Function: sub_B2AC90
// Address: 0xb2ac90
//
bool __fastcall sub_B2AC90(__int64 a1)
{
  __int64 v1; // rax
  int v2; // r12d
  __int64 v3; // r8
  bool result; // al

  v1 = sub_B2E500(a1);
  v2 = sub_B2A630(v1);
  v3 = sub_BA91D0(*(_QWORD *)(a1 + 40), "eh-asynch", 9);
  result = 0;
  if ( !v3 )
    return (unsigned int)(v2 - 7) > 1;
  return result;
}
