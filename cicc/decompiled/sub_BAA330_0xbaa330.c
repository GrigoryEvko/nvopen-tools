// Function: sub_BAA330
// Address: 0xbaa330
//
char __fastcall sub_BAA330(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi
  unsigned int v3; // ebx

  v1 = sub_BA91D0(a1, "DWARF64", 7u);
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 136);
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      LOBYTE(v1) = *(_QWORD *)(v2 + 24) == 1;
    else
      LOBYTE(v1) = v3 - 1 == (unsigned int)sub_C444A0(v2 + 24);
  }
  return v1;
}
