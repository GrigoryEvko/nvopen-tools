// Function: sub_27FF0F0
// Address: 0x27ff0f0
//
bool __fastcall sub_27FF0F0(__int64 a1, __int64 a2)
{
  bool result; // al
  _BYTE *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r12

  if ( !a2 )
    return 0;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 3 )
    return 0;
  v3 = *(_BYTE **)(a2 - 96);
  if ( *v3 != 82 )
    return 0;
  v4 = *((_QWORD *)v3 - 8);
  if ( !v4 || !*((_QWORD *)v3 - 4) )
    return 0;
  v5 = *(_QWORD *)(a2 - 32);
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(a2 - 64);
  if ( !v6 )
    return 0;
  result = sub_D97040(a1, *(_QWORD *)(v4 + 8));
  if ( result )
    return v5 != v6;
  return result;
}
