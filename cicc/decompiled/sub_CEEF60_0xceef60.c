// Function: sub_CEEF60
// Address: 0xceef60
//
__int64 __fastcall sub_CEEF60(__int64 **a1, __int64 a2)
{
  __int64 *v3; // r14
  const char *v4; // r13
  size_t v5; // rdx
  size_t v6; // r12
  int v7; // eax
  unsigned int v8; // r8d

  if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 4 <= 1 )
    return 1;
  v3 = *a1;
  v4 = sub_BD5D20(a2);
  v6 = v5;
  v7 = sub_C92610();
  LOBYTE(v8) = (unsigned int)sub_C92860(v3, v4, v6, v7) != -1;
  return v8;
}
