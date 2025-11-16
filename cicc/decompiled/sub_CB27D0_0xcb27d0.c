// Function: sub_CB27D0
// Address: 0xcb27d0
//
char __fastcall sub_CB27D0(__int64 a1, const void *a2, size_t a3)
{
  __int64 v3; // rax
  char *v4; // rax

  sub_CB1B10(a1, a2, a3);
  v3 = *(unsigned int *)(a1 + 40);
  if ( !(_DWORD)v3
    || (LOBYTE(v4) = sub_CB2090(*(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v3 - 4)), !(_BYTE)v4)
    && (LOBYTE(v4) = sub_CB27C0(*(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * *(unsigned int *)(a1 + 40) - 4)), !(_BYTE)v4) )
  {
    v4 = "\n";
    *(_QWORD *)(a1 + 104) = 1;
    *(_QWORD *)(a1 + 96) = "\n";
  }
  return (char)v4;
}
