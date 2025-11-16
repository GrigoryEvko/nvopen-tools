// Function: sub_396B840
// Address: 0x396b840
//
__int64 __fastcall sub_396B840(__int64 a1)
{
  unsigned int i; // r12d
  __int64 v2; // rbx
  _QWORD *v3; // rdi
  int v4; // eax

  if ( !a1 )
    return 0;
  i = 1;
  if ( *(_BYTE *)(a1 + 16) != 3 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    for ( i = 0; v2; i += v4 )
    {
      v3 = sub_1648700(v2);
      if ( *((_BYTE *)v3 + 16) >= 0x11u )
        v3 = 0;
      v4 = sub_396B840(v3);
      v2 = *(_QWORD *)(v2 + 8);
    }
  }
  return i;
}
