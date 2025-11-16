// Function: sub_117C3B0
// Address: 0x117c3b0
//
bool __fastcall sub_117C3B0(__int64 a1)
{
  unsigned int v1; // r12d
  int v2; // r13d
  int v3; // eax

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 > 0x40 )
  {
    v2 = sub_C44630(a1);
    if ( *(_DWORD *)(a1 + 24) <= 0x40u )
      goto LABEL_3;
LABEL_6:
    v3 = sub_C44630(a1 + 16);
    return v2 + v3 == v1;
  }
  v2 = sub_39FAC40(*(_QWORD *)a1);
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
    goto LABEL_6;
LABEL_3:
  v3 = sub_39FAC40(*(_QWORD *)(a1 + 16));
  return v2 + v3 == v1;
}
