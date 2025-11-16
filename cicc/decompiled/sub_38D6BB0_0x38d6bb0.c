// Function: sub_38D6BB0
// Address: 0x38d6bb0
//
bool __fastcall sub_38D6BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rax

  v5 = *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 )
    return *(_QWORD *)(a4 + 24) == *(_QWORD *)(v5 + 24);
  if ( (*(_BYTE *)(a3 + 9) & 0xC) == 8
    && (*(_BYTE *)(a3 + 8) |= 4u,
        v5 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a3 + 24)),
        *(_QWORD *)a3 = v5 | *(_QWORD *)a3 & 7LL,
        v5) )
  {
    return *(_QWORD *)(a4 + 24) == *(_QWORD *)(v5 + 24);
  }
  else
  {
    return *(_QWORD *)(a4 + 24) == 0;
  }
}
