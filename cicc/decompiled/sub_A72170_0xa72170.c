// Function: sub_A72170
// Address: 0xa72170
//
__int64 __fastcall sub_A72170(__int64 a1, const void *a2, __int64 a3)
{
  unsigned int v3; // r13d
  const void *v6; // rax
  size_t v7; // rdx

  v3 = 0;
  if ( *(_BYTE *)(a1 + 8) != 2 )
    return v3;
  v6 = (const void *)sub_A71FC0(a1);
  if ( v7 != a3 )
    return v3;
  v3 = 1;
  if ( !v7 )
    return v3;
  LOBYTE(v3) = memcmp(v6, a2, v7) == 0;
  return v3;
}
