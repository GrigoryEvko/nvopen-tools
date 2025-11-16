// Function: sub_84DA20
// Address: 0x84da20
//
__int64 __fastcall sub_84DA20(__int64 a1)
{
  __int64 result; // rax

  if ( a1 )
  {
    *(_DWORD *)(a1 + 4) = 1;
    *(_QWORD *)(a1 + 3112) = 0;
    *(_QWORD *)(a1 + 3120) = 0;
    *(_QWORD *)(a1 + 3104) = a1 + 8;
    return a1 + 8;
  }
  return result;
}
