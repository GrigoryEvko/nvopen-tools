// Function: sub_1EE61D0
// Address: 0x1ee61d0
//
bool __fastcall sub_1EE61D0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 184LL);
  if ( *(_BYTE *)(a1 + 56) )
    return (v1 & 0xFFFFFFFFFFFFFFF8LL) != 0;
  else
    return v1 == 0;
}
