// Function: sub_2F753A0
// Address: 0x2f753a0
//
bool __fastcall sub_2F753A0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 440LL);
  if ( *(_BYTE *)(a1 + 56) )
    return (v1 & 0xFFFFFFFFFFFFFFF8LL) != 0;
  else
    return v1 == 0;
}
