// Function: sub_255D820
// Address: 0x255d820
//
bool __fastcall sub_255D820(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  v2 = *(_QWORD *)(a1 + 88);
  if ( !v2 )
    v2 = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD *)(a2 + 88);
  if ( !v3 )
    v3 = *(unsigned int *)(a2 + 8);
  return v3 == v2 && sub_255D700(a1, a2);
}
