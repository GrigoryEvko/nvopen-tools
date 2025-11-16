// Function: sub_8C6230
// Address: 0x8c6230
//
_BOOL8 __fastcall sub_8C6230(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax
  __int64 v3; // rax
  __int64 *v4; // rcx
  __int64 v5; // rdx
  __int64 *v6; // rcx

  result = 0;
  if ( ((*(_BYTE *)(a1 + 81) ^ *(_BYTE *)(a2 + 81)) & 0x10) == 0 )
  {
    v3 = *(_QWORD *)(a1 + 64);
    v4 = *(__int64 **)(v3 + 32);
    if ( v4 )
      v3 = *v4;
    v5 = *(_QWORD *)(a2 + 64);
    v6 = *(__int64 **)(v5 + 32);
    if ( v6 )
      v5 = *v6;
    return v3 == v5;
  }
  return result;
}
