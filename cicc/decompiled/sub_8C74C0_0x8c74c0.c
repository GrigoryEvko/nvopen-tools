// Function: sub_8C74C0
// Address: 0x8c74c0
//
__int64 __fastcall sub_8C74C0(unsigned __int8 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = qword_4F60340[a1];
  if ( result )
  {
    v2 = *(_QWORD *)(result + 32);
    if ( v2 )
      return *(_QWORD *)v2;
  }
  return result;
}
