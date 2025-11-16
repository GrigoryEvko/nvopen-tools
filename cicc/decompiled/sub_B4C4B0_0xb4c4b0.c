// Function: sub_B4C4B0
// Address: 0xb4c4b0
//
__int64 __fastcall sub_B4C4B0(__int64 a1, __int64 a2)
{
  int v2; // r12d
  int v3; // eax
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // rdi

  v2 = *(_DWORD *)(a1 + 4);
  sub_B4C480(a1, 1u);
  v3 = *(_DWORD *)(a1 + 4);
  v4 = v2 & 0x7FFFFFF;
  v5 = (v3 + 1) & 0x7FFFFFF;
  result = (unsigned int)v5 | v3 & 0xF8000000;
  *(_DWORD *)(a1 + 4) = result;
  if ( (result & 0x40000000) != 0 )
    v7 = *(_QWORD *)(a1 - 8);
  else
    v7 = a1 - 32 * v5;
  v8 = v7 + 32 * v4;
  if ( *(_QWORD *)v8 )
  {
    result = *(_QWORD *)(v8 + 8);
    **(_QWORD **)(v8 + 16) = result;
    if ( result )
      *(_QWORD *)(result + 16) = *(_QWORD *)(v8 + 16);
  }
  *(_QWORD *)v8 = a2;
  if ( a2 )
  {
    result = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v8 + 8) = result;
    if ( result )
      *(_QWORD *)(result + 16) = v8 + 8;
    *(_QWORD *)(v8 + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v8;
  }
  return result;
}
