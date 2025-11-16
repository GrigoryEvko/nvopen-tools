// Function: sub_B5B640
// Address: 0xb5b640
//
bool __fastcall sub_B5B640(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // ecx

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v2 = *(_DWORD *)(v1 + 36) - 311;
  return v2 <= 0x1C && ((1LL << v2) & 0x18400003) != 0;
}
