// Function: sub_B91F50
// Address: 0xb91f50
//
__int64 __fastcall sub_B91F50(__int64 a1, const void *a2, size_t a3)
{
  __int64 *v4; // rax
  unsigned int v5; // eax

  v4 = (__int64 *)sub_BD5C60(a1, a2);
  v5 = sub_B6ED60(v4, a2, a3);
  if ( !v5 )
    return *(_QWORD *)(a1 + 48);
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    return sub_B91C10(a1, v5);
  return 0;
}
