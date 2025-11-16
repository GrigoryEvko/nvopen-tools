// Function: sub_170C140
// Address: 0x170c140
//
void __fastcall sub_170C140(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 v3; // rax

  v2 = *a2;
  sub_170B990(*a1, *a2);
  if ( *(_BYTE *)(v2 + 16) == 78 )
  {
    v3 = *(_QWORD *)(v2 - 24);
    if ( !*(_BYTE *)(v3 + 16) && *(_DWORD *)(v3 + 36) == 4 )
      sub_14CE830(a1[1], v2);
  }
}
