// Function: sub_38DD190
// Address: 0x38dd190
//
__int64 __fastcall sub_38DD190(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  result = sub_38DD140(a1);
  if ( result )
  {
    *(_QWORD *)(result + 16) = a2;
    *(_DWORD *)(result + 60) = a3;
  }
  return result;
}
