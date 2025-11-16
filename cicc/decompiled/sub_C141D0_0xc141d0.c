// Function: sub_C141D0
// Address: 0xc141d0
//
__int64 __fastcall sub_C141D0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 40) == v1 || !v1 )
    return 0;
  else
    return v1 - 48;
}
