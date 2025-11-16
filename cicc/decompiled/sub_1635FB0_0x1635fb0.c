// Function: sub_1635FB0
// Address: 0x1635fb0
//
const char *__fastcall sub_1635FB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rax
  const char *v5; // r8

  v2 = *(_QWORD *)(a1 + 16);
  v3 = sub_163A1D0(a1, a2);
  v4 = sub_163A340(v3, v2);
  v5 = "Unnamed pass: implement Pass::getPassName()";
  if ( v4 )
    return *(const char **)v4;
  return v5;
}
