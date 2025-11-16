// Function: sub_B4BEA0
// Address: 0xb4bea0
//
__int64 __fastcall sub_B4BEA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // rax

  if ( a3 )
    *(_WORD *)(a1 + 2) |= 1u;
  result = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)result )
  {
    v4 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = a2;
  if ( a2 )
  {
    v5 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(result + 8) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = result + 8;
    *(_QWORD *)(result + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = result;
  }
  if ( a3 )
  {
    v6 = 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + a1;
    if ( *(_QWORD *)v6 )
    {
      v7 = *(_QWORD *)(v6 + 8);
      **(_QWORD **)(v6 + 16) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = *(_QWORD *)(v6 + 16);
    }
    *(_QWORD *)v6 = a3;
    result = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v6 + 8) = result;
    if ( result )
      *(_QWORD *)(result + 16) = v6 + 8;
    *(_QWORD *)(v6 + 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = v6;
  }
  return result;
}
