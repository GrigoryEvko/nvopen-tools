// Function: sub_AC42A0
// Address: 0xac42a0
//
__int64 __fastcall sub_AC42A0(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 v3; // rax
  __int64 result; // rax

  sub_BD35F0(a1, *(_QWORD *)(a2 + 8), 7);
  v2 = *(_QWORD *)(a1 - 32) == 0;
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0x38000000 | 1;
  if ( !v2 )
  {
    v3 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v3;
    if ( v3 )
      *(_QWORD *)(v3 + 16) = *(_QWORD *)(a1 - 16);
  }
  result = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 - 32) = a2;
  *(_QWORD *)(a1 - 24) = result;
  if ( result )
    *(_QWORD *)(result + 16) = a1 - 24;
  *(_QWORD *)(a1 - 16) = a2 + 16;
  *(_QWORD *)(a2 + 16) = a1 - 32;
  return result;
}
