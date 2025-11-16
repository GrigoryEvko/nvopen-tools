// Function: sub_B59D40
// Address: 0xb59d40
//
__int64 __fastcall sub_B59D40(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = a1;
  v3 = a1 + 32 * (4LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v3 )
  {
    result = *(_QWORD *)(v3 + 8);
    **(_QWORD **)(v3 + 16) = result;
    if ( result )
      *(_QWORD *)(result + 16) = *(_QWORD *)(v3 + 16);
  }
  *(_QWORD *)v3 = a2;
  if ( a2 )
  {
    result = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v3 + 8) = result;
    if ( result )
      *(_QWORD *)(result + 16) = v3 + 8;
    *(_QWORD *)(v3 + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v3;
  }
  return result;
}
