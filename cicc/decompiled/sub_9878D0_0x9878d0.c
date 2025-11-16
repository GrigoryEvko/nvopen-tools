// Function: sub_9878D0
// Address: 0x9878d0
//
__int64 __fastcall sub_9878D0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = a2;
    return sub_C43690(a1 + 16, 0, 0);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = 0;
  }
  return result;
}
