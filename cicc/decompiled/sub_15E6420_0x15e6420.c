// Function: sub_15E6420
// Address: 0x15e6420
//
void __fastcall sub_15E6420(__int64 a1, __int64 a2)
{
  const void *v2; // rax
  size_t v3; // rdx

  sub_15E4BE0(a1, a2);
  sub_15E4CC0(a1, (unsigned int)(1 << (*(_DWORD *)(a2 + 32) >> 15)) >> 1);
  if ( (*(_BYTE *)(a2 + 34) & 0x20) != 0 )
  {
    v2 = (const void *)sub_15E61A0(a2);
    sub_15E5D20(a1, v2, v3);
  }
  else
  {
    sub_15E5D20(a1, 0, 0);
  }
}
