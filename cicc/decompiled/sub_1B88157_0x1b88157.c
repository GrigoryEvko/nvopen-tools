// Function: sub_1B88157
// Address: 0x1b88157
//
void __fastcall sub_1B88157()
{
  __int64 v0; // rbp
  __int64 v1; // rsi
  __int64 v2; // rdx

  v1 = *(_QWORD *)(v0 - 112);
  v2 = ~(1LL << (*(_BYTE *)(v0 - 216) - 1));
  if ( *(_DWORD *)(v0 - 104) > 0x40u )
    *(_QWORD *)(v1 + 8LL * ((unsigned int)(*(_DWORD *)(v0 - 216) - 1) >> 6)) &= v2;
  else
    *(_QWORD *)(v0 - 112) = v1 & v2;
  JUMPOUT(0x1B87F9D);
}
