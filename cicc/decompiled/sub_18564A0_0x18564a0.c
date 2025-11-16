// Function: sub_18564A0
// Address: 0x18564a0
//
__int64 __fastcall sub_18564A0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 24) == v1 || !v1 )
    return 0;
  else
    return v1 - 56;
}
