// Function: sub_AE94B0
// Address: 0xae94b0
//
__int64 __fastcall sub_AE94B0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rdi
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 8);
  v2 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v1 & 4) != 0 )
    v2 = (_QWORD *)*v2;
  result = sub_B9F8A0(v2);
  if ( result )
    return *(_QWORD *)(result + 16);
  return result;
}
