// Function: sub_B54850
// Address: 0xb54850
//
__int64 __fastcall sub_B54850(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // rdi

  v2 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( (unsigned int)(v2 + 1) > *(_DWORD *)(a1 + 72) )
    sub_B54680(a1);
  v3 = ((_DWORD)v2 + 1) & 0x7FFFFFF;
  result = (unsigned int)v3 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  *(_DWORD *)(a1 + 4) = result;
  if ( (result & 0x40000000) != 0 )
    v5 = *(_QWORD *)(a1 - 8);
  else
    v5 = a1 - 32 * v3;
  v6 = v5 + 32 * v2;
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
