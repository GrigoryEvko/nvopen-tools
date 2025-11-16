// Function: sub_17570E0
// Address: 0x17570e0
//
__int64 __fastcall sub_17570E0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax

  v2 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v2 )
    return 0;
  **a1 = v2;
  v3 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( !v3 )
    return 0;
  *a1[1] = v3;
  return 1;
}
