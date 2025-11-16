// Function: sub_B4C570
// Address: 0xb4c570
//
__int64 __fastcall sub_B4C570(__int64 a1, __int64 a2)
{
  __int64 i; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 result; // rax

  for ( i = *(_QWORD *)(a1 - 8) + 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) - 32; i != a2; a2 += 32 )
  {
    v3 = *(_QWORD *)(a2 + 32);
    if ( *(_QWORD *)a2 )
    {
      v4 = *(_QWORD *)(a2 + 8);
      **(_QWORD **)(a2 + 16) = v4;
      if ( v4 )
        *(_QWORD *)(v4 + 16) = *(_QWORD *)(a2 + 16);
    }
    *(_QWORD *)a2 = v3;
    if ( v3 )
    {
      v5 = *(_QWORD *)(v3 + 16);
      *(_QWORD *)(a2 + 8) = v5;
      if ( v5 )
        *(_QWORD *)(v5 + 16) = a2 + 8;
      *(_QWORD *)(a2 + 16) = v3 + 16;
      *(_QWORD *)(v3 + 16) = a2;
    }
  }
  if ( *(_QWORD *)i )
  {
    v6 = *(_QWORD *)(i + 8);
    **(_QWORD **)(i + 16) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(i + 16);
  }
  *(_QWORD *)i = 0;
  result = (*(_DWORD *)(a1 + 4) + 0x7FFFFFF) & 0x7FFFFFF | *(_DWORD *)(a1 + 4) & 0xF8000000;
  *(_DWORD *)(a1 + 4) = result;
  return result;
}
