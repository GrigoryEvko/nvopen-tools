// Function: sub_B99460
// Address: 0xb99460
//
void __fastcall sub_B99460(__int64 a1, const void *a2, size_t a3, __int64 a4)
{
  __int64 *v6; // rax
  unsigned int v7; // eax

  if ( a4 || (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v6 = (__int64 *)sub_BD5C60(a1, a2);
    v7 = sub_B6ED60(v6, a2, a3);
    sub_B99110(a1, v7, a4);
  }
}
