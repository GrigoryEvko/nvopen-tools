// Function: sub_7D8A60
// Address: 0x7d8a60
//
__int64 __fastcall sub_7D8A60(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // r12

  result = dword_4D047EC;
  if ( dword_4D047EC )
  {
    v2 = *a1;
    result = sub_8E3210(*a1);
    if ( (_DWORD)result )
      return sub_8DD360(v2);
  }
  return result;
}
