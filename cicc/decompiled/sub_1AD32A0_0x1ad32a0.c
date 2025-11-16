// Function: sub_1AD32A0
// Address: 0x1ad32a0
//
__int64 __fastcall sub_1AD32A0(__int64 a1)
{
  __int64 v1; // rbx
  _QWORD *v2; // rax
  __int64 v3; // rax

  if ( !a1 )
    return 0;
  v1 = a1;
  do
  {
    v2 = sub_1648700(v1);
    if ( *((_BYTE *)v2 + 16) == 78 )
    {
      v3 = *(v2 - 3);
      if ( !*(_BYTE *)(v3 + 16) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v3 + 36) - 116) <= 1 )
        return 1;
    }
    v1 = *(_QWORD *)(v1 + 8);
  }
  while ( v1 );
  return 0;
}
