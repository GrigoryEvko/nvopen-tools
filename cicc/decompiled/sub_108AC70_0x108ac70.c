// Function: sub_108AC70
// Address: 0x108ac70
//
__int64 __fastcall sub_108AC70(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  __int64 v5; // rdi

  v2 = 0;
  if ( *(_BYTE *)(*a2 + 2376) )
    return v2;
  v4 = sub_1089C70(*(_QWORD *)(a1 + 112), a2);
  v5 = *(_QWORD *)(a1 + 120);
  v2 = v4;
  if ( !v5 )
    return v2;
  else
    return sub_1089C70(v5, a2) + v4;
}
