// Function: sub_1FDAF60
// Address: 0x1fdaf60
//
__int64 __fastcall sub_1FDAF60(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = a1[11];
  if ( *(_DWORD *)(v2 + 504) != 32 || *(_DWORD *)(v2 + 516) != 9 )
    return 1;
  sub_1FDACC0(a1, a2);
  return 1;
}
