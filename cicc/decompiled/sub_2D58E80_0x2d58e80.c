// Function: sub_2D58E80
// Address: 0x2d58e80
//
__int64 __fastcall sub_2D58E80(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_QWORD *)(a1 + 16);
  if ( (*(_BYTE *)(v1 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(v1 - 8);
  else
    v3 = v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF);
  result = 32LL * *(unsigned int *)(a1 + 24) + v3;
  if ( *(_QWORD *)result )
  {
    v5 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = v2;
  if ( v2 )
  {
    v6 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(result + 8) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = result + 8;
    *(_QWORD *)(result + 16) = v2 + 16;
    *(_QWORD *)(v2 + 16) = result;
  }
  return result;
}
