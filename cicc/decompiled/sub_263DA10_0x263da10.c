// Function: sub_263DA10
// Address: 0x263da10
//
__int64 __fastcall sub_263DA10(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 24) == v1 )
    return 0;
  *(_QWORD *)(a1 + 8) = v1 + 16;
  return 1;
}
