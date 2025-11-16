// Function: sub_1370F90
// Address: 0x1370f90
//
__int64 __fastcall sub_1370F90(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rax
  __int64 v3; // r8

  v2 = *a2;
  v3 = 0;
  if ( (_DWORD)v2 != -1 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 8) + 24 * v2);
  return v3;
}
