// Function: sub_1EE6200
// Address: 0x1ee6200
//
bool __fastcall sub_1EE6200(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 192LL);
  if ( *(_BYTE *)(a1 + 56) )
    return (v1 & 0xFFFFFFFFFFFFFFF8LL) != 0;
  else
    return v1 == 0;
}
