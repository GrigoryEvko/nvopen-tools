// Function: sub_8282E0
// Address: 0x8282e0
//
void __fastcall sub_8282E0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  char v10; // di
  unsigned int v11; // esi

  if ( (*(_BYTE *)(a1 + 13) & 4) != 0 )
  {
    v10 = 8;
    if ( dword_4F077BC )
      v10 = (_DWORD)qword_4F077B4 == 0 ? 5 : 8;
    sub_6E5C80(v10, 0x343u, a3);
  }
  v11 = *(_DWORD *)(a1 + 8);
  if ( v11 )
  {
    if ( sub_6E53E0(5, v11, a3) )
      sub_6858D0(*(_DWORD *)(a1 + 8), a3, a4, a5);
    *(_DWORD *)(a2 + 32) = 0;
  }
}
