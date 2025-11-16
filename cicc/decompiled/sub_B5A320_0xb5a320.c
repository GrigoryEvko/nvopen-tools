// Function: sub_B5A320
// Address: 0xb5a320
//
__int64 __fastcall sub_B5A320(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // rdi

  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v4 = (unsigned int)sub_B5A1E0(*(_DWORD *)(v2 + 36));
  result = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v6 = a1 + 32 * (v4 - result);
  if ( *(_QWORD *)v6 )
  {
    result = *(_QWORD *)(v6 + 8);
    **(_QWORD **)(v6 + 16) = result;
    if ( result )
      *(_QWORD *)(result + 16) = *(_QWORD *)(v6 + 16);
  }
  *(_QWORD *)v6 = a2;
  if ( a2 )
  {
    result = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v6 + 8) = result;
    if ( result )
      *(_QWORD *)(result + 16) = v6 + 8;
    *(_QWORD *)(v6 + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v6;
  }
  return result;
}
