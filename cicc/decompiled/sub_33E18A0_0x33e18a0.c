// Function: sub_33E18A0
// Address: 0x33e18a0
//
_BOOL8 __fastcall sub_33E18A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rbx

  v6 = sub_33E1790(a2, a3, 1u, a4, a5, a6);
  if ( !v6 )
    return *(_DWORD *)(a2 + 24) == 245;
  v7 = *(_QWORD *)(v6 + 96);
  if ( *(void **)(v7 + 24) == sub_C33340() )
    v8 = *(_QWORD *)(v7 + 32);
  else
    v8 = v7 + 24;
  return ((*(_BYTE *)(v8 + 20) >> 3) ^ 1) & 1;
}
