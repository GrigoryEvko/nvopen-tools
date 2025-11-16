// Function: sub_33E2470
// Address: 0x33e2470
//
__int64 __fastcall sub_33E2470(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 result; // rax
  __int64 v4; // rax
  int v5; // edx

  v2 = *(_DWORD *)(a2 + 24);
  if ( v2 == 12 || v2 == 36 )
    return 1;
  result = sub_33CA720(a2);
  if ( (_BYTE)result )
    return 1;
  if ( *(_DWORD *)(a2 + 24) == 168 )
  {
    v4 = **(_QWORD **)(a2 + 40);
    v5 = *(_DWORD *)(v4 + 24);
    LOBYTE(v4) = v5 == 36;
    LOBYTE(v5) = v5 == 12;
    return v5 | (unsigned int)v4;
  }
  return result;
}
