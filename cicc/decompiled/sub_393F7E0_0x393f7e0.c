// Function: sub_393F7E0
// Address: 0x393f7e0
//
__int64 __fastcall sub_393F7E0(__int64 a1)
{
  unsigned int v1; // r12d
  _QWORD *v2; // rbx

  v1 = 0;
  v2 = *(_QWORD **)(a1 + 8);
  if ( v2 && strlen(*(const char **)(a1 + 8)) == 8 )
    LOBYTE(v1) = *v2 == 0x3430372A67636461LL;
  return v1;
}
