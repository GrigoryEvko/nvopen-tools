// Function: sub_17330B0
// Address: 0x17330b0
//
__int64 __fastcall sub_17330B0(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  if ( *(_BYTE *)(a2 + 16) != 79 )
    return 0;
  v3 = *(_QWORD *)(a2 - 72);
  if ( !v3 )
    return 0;
  **a1 = v3;
  v4 = *(_QWORD *)(a2 - 48);
  if ( !v4 )
    return 0;
  *a1[1] = v4;
  v5 = *(_QWORD *)(a2 - 24);
  if ( !v5 )
    return 0;
  *a1[2] = v5;
  return 1;
}
