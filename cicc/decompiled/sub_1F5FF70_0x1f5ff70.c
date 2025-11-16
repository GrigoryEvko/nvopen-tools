// Function: sub_1F5FF70
// Address: 0x1f5ff70
//
__int64 __fastcall sub_1F5FF70(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( !v1 )
    return v1;
  while ( 1 )
  {
    v2 = sub_1648700(v1);
    if ( *((_BYTE *)v2 + 16) == 32 )
      break;
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return v1;
  }
  v1 = 0;
  if ( (*((_BYTE *)v2 + 18) & 1) != 0 )
    return v2[3 * (1LL - (*((_DWORD *)v2 + 5) & 0xFFFFFFF))];
  else
    return v1;
}
