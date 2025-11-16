// Function: sub_38DD1C0
// Address: 0x38dd1c0
//
__int64 __fastcall sub_38DD1C0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  result = sub_38DD140(a1);
  if ( result )
  {
    *(_QWORD *)(result + 24) = a2;
    *(_DWORD *)(result + 64) = a3;
  }
  return result;
}
