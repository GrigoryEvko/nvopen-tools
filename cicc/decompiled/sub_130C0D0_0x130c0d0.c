// Function: sub_130C0D0
// Address: 0x130c0d0
//
__int64 __fastcall sub_130C0D0(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi

  v2 = a1 + 58648;
  v3 = a1 + 60432;
  if ( a2 == 1 )
    v3 = v2;
  return *(_QWORD *)(v3 + 120);
}
