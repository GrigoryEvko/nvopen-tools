// Function: sub_21CF340
// Address: 0x21cf340
//
bool __fastcall sub_21CF340(__int64 a1)
{
  int v1; // eax

  v1 = *(_DWORD *)(*(_QWORD *)(a1 + 81552) + 82316LL);
  if ( v1 == -1 )
    return ((*(_BYTE *)(*(_QWORD *)(a1 + 8) + 792LL) >> 1) ^ 1) & 1;
  else
    return v1 == 1;
}
