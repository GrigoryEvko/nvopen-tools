// Function: sub_8C7470
// Address: 0x8c7470
//
__int64 __fastcall sub_8C7470(unsigned __int8 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = qword_4F603C0[a1];
  if ( result )
  {
    v2 = *(_QWORD *)(result + 32);
    if ( v2 )
      return *(_QWORD *)v2;
  }
  return result;
}
